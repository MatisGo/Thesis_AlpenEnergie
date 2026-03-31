"""
mode_water_value.py
===================
WATER_VALUE mode — LP that respects the forecast while optimising timing.

Core idea
---------
Every mm of water in a reservoir has an *opportunity cost*: the revenue it
could generate if used at the best future price.  At each timestep the LP
must decide whether the current price is worth consuming that water, or
whether to save it for a more profitable hour later.

This is implemented as a rolling 24-hour LP with two objectives:

  1. Revenue    — same as PRICE_ARBITRAGE (maximise spot revenue).
  2. Forecast   — soft penalty proportional to |production − forecast_kW|.

FORECAST_WEIGHT controls the trade-off:
  • High weight → sticks closely to the forecast regardless of prices.
  • Low weight  → behaves like pure price arbitrage.
  • The recommended default (0.03 CHF/kW) makes forecast adherence worth
    roughly 3× the average 5-min energy revenue, so the model follows the
    forecast except when prices differ significantly from the daily average.

The thresholds from FORECAST mode disappear entirely.  The LP naturally
reduces production when reservoir levels are low because low levels raise
the shadow price of water above the current spot price.
"""

import sys
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals, Reals,
    Constraint, Objective, maximize, value, SolverFactory,
)
from hydro_constants import (
    P_MAX_M2, P_MAX_M1, RAMP_MAX, RAMP_WINDOW, TIMESTEP_HOURS,
    BIDMI_LEVEL_MIN, BIDMI_LEVEL_MAX, HASELHOLZ_LEVEL_MIN, HASELHOLZ_LEVEL_MAX,
    BIDMI_RANGE, HASELHOLZ_RANGE,
    BIDMI_LS_PER_MM, HASELHOLZ_LS_PER_MM,
    COEFF_M2_BIDMI, COEFF_M2_CASCADE, COEFF_M1_HASELHOLZ,
    water_balance_step, spill_to_kwh, attach_common_results,
)
from mode_price_arbitrage import (
    get_solver,
    TERMINAL_FREE_ZONE, PW_ZONE2_WIDTH, PW_ZONE3_WIDTH,
    PW_SLOPE2, PW_SLOPE3, PW_SLOPE4,
    SPILL_PENALTY, INTRADAY_FLOOR_PENALTY,
    RECOVERY_PENALTY_MM_B, RECOVERY_PENALTY_MM_H, THRESHOLD_B, THRESHOLD_H,
)

# ---------------------------------------------------------------------------
# Forecast tracking weight
# ---------------------------------------------------------------------------
# CHF per kW of deviation from forecast per 5-min timestep.
# At 100 CHF/MWh avg price: 1 kW earns 100 × (5/60) / 1000 = 0.0083 CHF.
# FORECAST_WEIGHT = 0.03 → forecast adherence is worth ~3× average revenue.
FORECAST_WEIGHT = 0.03   # CHF/kW


# ---------------------------------------------------------------------------
# LP model builder
# ---------------------------------------------------------------------------

def _build_model(N, forecast, demand, price, inflow_b, inflow_h,
                 lb0, lh0, target_lb, target_lh, floor_lb, floor_lh):
    m     = ConcreteModel()
    T     = list(range(N))
    T_ext = list(range(N + 1))
    m.T     = Set(initialize=T)
    m.T_ext = Set(initialize=T_ext)

    # Decision variables
    m.P_M2 = Var(m.T, domain=NonNegativeReals, bounds=(0.0, P_MAX_M2))
    m.P_M1 = Var(m.T, domain=NonNegativeReals, bounds=(0.0, P_MAX_M1))

    # Reservoir levels
    m.LB = Var(m.T_ext, bounds=(BIDMI_LEVEL_MIN, None))
    m.LH = Var(m.T_ext, bounds=(HASELHOLZ_LEVEL_MIN, None))
    m.LB[0].fix(lb0);  m.LH[0].fix(lh0)

    m.lb_hi = Constraint([t for t in T_ext if t >= 1],
                         rule=lambda m, t: m.LB[t] <= BIDMI_LEVEL_MAX)
    m.lh_hi = Constraint([t for t in T_ext if t >= 1],
                         rule=lambda m, t: m.LH[t] <= HASELHOLZ_LEVEL_MAX)

    # Spillage
    m.spill_b = Var(m.T, domain=NonNegativeReals)
    m.spill_h = Var(m.T, domain=NonNegativeReals)

    # Ramp constraints
    def _ramp(m, t, var, direction):
        if t < RAMP_WINDOW: return Constraint.Skip
        diff = var[t] - var[t - RAMP_WINDOW] if direction == 'up' else var[t - RAMP_WINDOW] - var[t]
        return diff <= RAMP_MAX

    m.ramp_m2_up = Constraint(m.T, rule=lambda m, t: _ramp(m, t, m.P_M2, 'up'))
    m.ramp_m2_dn = Constraint(m.T, rule=lambda m, t: _ramp(m, t, m.P_M2, 'dn'))
    m.ramp_m1_up = Constraint(m.T, rule=lambda m, t: _ramp(m, t, m.P_M1, 'up'))
    m.ramp_m1_dn = Constraint(m.T, rule=lambda m, t: _ramp(m, t, m.P_M1, 'dn'))

    # Water balance
    m.wb_b = Constraint(m.T, rule=lambda m, t:
        m.LB[t+1] == m.LB[t] + inflow_b[t] / BIDMI_LS_PER_MM
                     - COEFF_M2_BIDMI * m.P_M2[t] - m.spill_b[t])
    m.wb_h = Constraint(m.T, rule=lambda m, t:
        m.LH[t+1] == m.LH[t] + inflow_h[t] / HASELHOLZ_LS_PER_MM
                     + COEFF_M2_CASCADE * m.P_M2[t] - COEFF_M1_HASELHOLZ * m.P_M1[t] - m.spill_h[t])

    # Intra-day soft floor
    m.floor_slack_b = Var(m.T, domain=NonNegativeReals)
    m.floor_slack_h = Var(m.T, domain=NonNegativeReals)
    m.floor_con_b = Constraint(m.T, rule=lambda m, t: m.floor_slack_b[t] >= floor_lb[t] - m.LB[t])
    m.floor_con_h = Constraint(m.T, rule=lambda m, t: m.floor_slack_h[t] >= floor_lh[t] - m.LH[t])

    # Piecewise terminal penalty (reuse from price_arbitrage)
    m.dev_b = Var(domain=NonNegativeReals); m.dev_b_con = Constraint(expr=m.dev_b >= target_lb - m.LB[N])
    m.dev_h = Var(domain=NonNegativeReals); m.dev_h_con = Constraint(expr=m.dev_h >= target_lh - m.LH[N])

    m.pw_b2 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE2_WIDTH))
    m.pw_b3 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE3_WIDTH))
    m.pw_b4 = Var(domain=NonNegativeReals)
    m.pw_b_link = Constraint(expr=m.pw_b2 + m.pw_b3 + m.pw_b4 >= m.dev_b - TERMINAL_FREE_ZONE)

    m.pw_h2 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE2_WIDTH))
    m.pw_h3 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE3_WIDTH))
    m.pw_h4 = Var(domain=NonNegativeReals)
    m.pw_h_link = Constraint(expr=m.pw_h2 + m.pw_h3 + m.pw_h4 >= m.dev_h - TERMINAL_FREE_ZONE)

    # Recovery soft penalty (same calibration as price_arbitrage)
    m.deficit_b = Var(m.T, domain=NonNegativeReals)
    m.deficit_h = Var(m.T, domain=NonNegativeReals)
    m.deficit_b_con = Constraint(m.T, rule=lambda m, t: m.deficit_b[t] >= THRESHOLD_B - m.LB[t])
    m.deficit_h_con = Constraint(m.T, rule=lambda m, t: m.deficit_h[t] >= THRESHOLD_H - m.LH[t])

    # ── Forecast tracking: linearised |production − forecast| ────────────
    # dev_pos[t] - dev_neg[t] = production[t] - forecast[t]
    # penalty = FORECAST_WEIGHT × (dev_pos + dev_neg)
    m.dev_pos = Var(m.T, domain=NonNegativeReals)
    m.dev_neg = Var(m.T, domain=NonNegativeReals)
    m.forecast_con = Constraint(m.T, rule=lambda m, t:
        m.dev_pos[t] - m.dev_neg[t] == m.P_M2[t] + m.P_M1[t] - forecast[t])

    # Objective
    revenue = sum(
        (m.P_M2[t] + m.P_M1[t] - demand[t]) * TIMESTEP_HOURS * price[t] / 1000.0
        for t in m.T)
    penalties = (
        - PW_SLOPE2 * (m.pw_b2 + m.pw_h2)
        - PW_SLOPE3 * (m.pw_b3 + m.pw_h3)
        - PW_SLOPE4 * (m.pw_b4 + m.pw_h4)
        - SPILL_PENALTY * (sum(m.spill_b[t] for t in m.T) + sum(m.spill_h[t] for t in m.T))
        - INTRADAY_FLOOR_PENALTY * (
            sum(m.floor_slack_b[t] for t in m.T) + sum(m.floor_slack_h[t] for t in m.T))
        - RECOVERY_PENALTY_MM_B * sum(m.deficit_b[t] for t in m.T)
        - RECOVERY_PENALTY_MM_H * sum(m.deficit_h[t] for t in m.T)
        - FORECAST_WEIGHT * sum(m.dev_pos[t] + m.dev_neg[t] for t in m.T)
    )
    m.obj = Objective(expr=revenue + penalties, sense=maximize)
    return m


# ---------------------------------------------------------------------------
# Day dispatch
# ---------------------------------------------------------------------------

def dispatch_day(day_df, lb0, lh0, target_lb, target_lh,
                 solver=None, floor_lb=None, floor_lh=None, **_kwargs):
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    forecast = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]
    demand   = day_df['Consumption_kW'].tolist()
    price    = day_df['Spot_Price_CHF_MWh'].tolist()
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    def _fit(arr, fill):
        if arr is None: return [fill] * N
        arr = list(arr)[:N];  arr += [fill] * (N - len(arr))
        return arr

    floor_lb = _fit(floor_lb, BIDMI_LEVEL_MIN)
    floor_lh = _fit(floor_lh, HASELHOLZ_LEVEL_MIN)

    model  = _build_model(N, forecast, demand, price, inflow_b, inflow_h,
                          lb0, lh0, target_lb, target_lh, floor_lb, floor_lh)
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    if status not in ('optimal', 'feasible'):
        print(f"  WARNING: solver status '{status}' on {day_df.loc[0,'DateTime'].date()} — filling with zeros")
        zeros = [0.0] * N
        attach_common_results(
            day_df, zeros, zeros, [lb0] * N, [lh0] * N, zeros, zeros,
            demand, price, target_lb, target_lh, mode_name='WATER_VALUE',
        )
        day_df['Forecast_Drift_kW'] = [-f for f in forecast]
        return day_df

    opt_m2      = [value(model.P_M2[t])     for t in range(N)]
    opt_m1      = [value(model.P_M1[t])     for t in range(N)]
    opt_lb      = [value(model.LB[t])       for t in range(N)]
    opt_lh      = [value(model.LH[t])       for t in range(N)]
    opt_spill_b = [value(model.spill_b[t])  for t in range(N)]
    opt_spill_h = [value(model.spill_h[t])  for t in range(N)]

    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, price, target_lb, target_lh,
        mode_name='WATER_VALUE',
    )

    day_df['Forecast_Drift_kW'] = [
        (opt_m2[t] + opt_m1[t]) - forecast[t] for t in range(N)]

    return day_df
