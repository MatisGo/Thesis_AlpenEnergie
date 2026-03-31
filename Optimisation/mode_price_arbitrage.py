"""
mode_price_arbitrage.py
=======================
PRICE_ARBITRAGE mode — pure LP profit maximisation.

The LP maximises energy trading revenue (spot price × net production)
subject to water balance, turbine limits, ramp rates, and reservoir bounds.

Recovery is modelled as a soft LP penalty: a slack variable captures how
far each reservoir is below 20% of its physical range.  The penalty is
calibrated so that producing 1 kW when the reservoir is at the threshold
costs approximately RECOVERY_PENALTY CHF — well above typical spot revenue.

A piecewise-linear terminal penalty keeps end-of-day levels near the
30-day seasonal average.  An intra-day soft floor (very gentle) guides the
shape of the level curve.
"""

import sys
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals,
    Constraint, Objective, maximize, value, SolverFactory,
)
from hydro_constants import (
    P_MAX_M2, P_MAX_M1, RAMP_MAX, RAMP_WINDOW, TIMESTEP_HOURS,
    BIDMI_LEVEL_MIN, BIDMI_LEVEL_MAX, HASELHOLZ_LEVEL_MIN, HASELHOLZ_LEVEL_MAX,
    BIDMI_RANGE, HASELHOLZ_RANGE,
    BIDMI_LS_PER_MM, HASELHOLZ_LS_PER_MM,
    COEFF_M2_BIDMI, COEFF_M2_CASCADE, COEFF_M1_HASELHOLZ,
    TURBINE_CASCADE_RATIO,
    water_balance_step, spill_to_kwh, attach_common_results,
)

# ---------------------------------------------------------------------------
# LP parameters
# ---------------------------------------------------------------------------

# Piecewise-linear terminal penalty (below seasonal target only)
TERMINAL_FREE_ZONE = 100.0   # mm free zone — no penalty
PW_ZONE2_WIDTH     = 100.0
PW_ZONE3_WIDTH     = 100.0
PW_SLOPE2          =   2.0   # CHF/mm
PW_SLOPE3          =   8.0   # CHF/mm
PW_SLOPE4          =  25.0   # CHF/mm

SPILL_PENALTY          = 1.0    # CHF/mm
INTRADAY_FLOOR_PENALTY = 0.05   # CHF/mm per timestep (gentle intra-day shape guidance)

# Recovery soft penalty
RECOVERY_THRESHOLD_PCT = 0.20
THRESHOLD_B = BIDMI_LEVEL_MIN     + RECOVERY_THRESHOLD_PCT * BIDMI_RANGE
THRESHOLD_H = HASELHOLZ_LEVEL_MIN + RECOVERY_THRESHOLD_PCT * HASELHOLZ_RANGE
RECOVERY_PENALTY       = 1.0                                  # CHF/kW equivalent
RECOVERY_PENALTY_MM_B  = RECOVERY_PENALTY * 4.537 * 8.33     # CHF/mm (~37.8)
RECOVERY_PENALTY_MM_H  = RECOVERY_PENALTY * 1.634 * 4.39     # CHF/mm (~7.2)


# ---------------------------------------------------------------------------
# LP model builder
# ---------------------------------------------------------------------------

def _build_model(N, demand, price, inflow_b, inflow_h,
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

    # Ramp constraints (200 kW per 15 min)
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

    # Piecewise terminal penalty
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

    # Recovery soft penalty
    m.deficit_b = Var(m.T, domain=NonNegativeReals)
    m.deficit_h = Var(m.T, domain=NonNegativeReals)
    m.deficit_b_con = Constraint(m.T, rule=lambda m, t: m.deficit_b[t] >= THRESHOLD_B - m.LB[t])
    m.deficit_h_con = Constraint(m.T, rule=lambda m, t: m.deficit_h[t] >= THRESHOLD_H - m.LH[t])

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
    )
    m.obj = Objective(expr=revenue + penalties, sense=maximize)
    return m


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def get_solver():
    solver = SolverFactory('glpk')
    if not solver.available():
        print("ERROR: GLPK not found.  Install via: conda install glpk")
        sys.exit(1)
    return solver


# ---------------------------------------------------------------------------
# Day dispatch
# ---------------------------------------------------------------------------

def dispatch_day(day_df, lb0, lh0, target_lb, target_lh,
                 solver=None, floor_lb=None, floor_lh=None, **_kwargs):
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    demand   = day_df['Consumption_kW'].tolist()
    price    = day_df['Spot_Price_CHF_MWh'].tolist()
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    # Pad / truncate floor arrays to exactly N
    def _fit(arr, fill):
        if arr is None: return [fill] * N
        arr = list(arr)[:N];  arr += [fill] * (N - len(arr))
        return arr

    floor_lb = _fit(floor_lb, BIDMI_LEVEL_MIN)
    floor_lh = _fit(floor_lh, HASELHOLZ_LEVEL_MIN)

    model  = _build_model(N, demand, price, inflow_b, inflow_h,
                          lb0, lh0, target_lb, target_lh, floor_lb, floor_lh)
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    if status not in ('optimal', 'feasible'):
        print(f"  WARNING: solver status '{status}' on {day_df.loc[0,'DateTime'].date()} — filling with zeros")
        zeros = [0.0] * N
        attach_common_results(
            day_df, zeros, zeros, [lb0] * N, [lh0] * N, zeros, zeros,
            demand, price, target_lb, target_lh, mode_name='PRICE_ARBITRAGE',
        )
        return day_df

    opt_m2     = [value(model.P_M2[t])  for t in range(N)]
    opt_m1     = [value(model.P_M1[t])  for t in range(N)]
    opt_lb     = [value(model.LB[t])    for t in range(N)]
    opt_lh     = [value(model.LH[t])    for t in range(N)]
    opt_spill_b = [value(model.spill_b[t]) for t in range(N)]
    opt_spill_h = [value(model.spill_h[t]) for t in range(N)]

    return attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, price, target_lb, target_lh,
        mode_name='PRICE_ARBITRAGE',
    )
