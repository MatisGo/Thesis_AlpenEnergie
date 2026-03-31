"""
mode_water_value.py
===================
WATER_VALUE mode — LP that minimises intraday imbalance costs while respecting
physical constraints.

Pricing structure
-----------------
The plant submits the CNN/LSTM forecast as its day-ahead bid.  Settlement:

  DA revenue  = (Forecast - Consumption) × DA_price × dt / 1000   [EUR]
              → Fixed at scheduling time; the LP cannot influence it.

  ID cost     = |Production - Forecast| × ID_price × dt / 1000    [EUR]
              → Always a cost: both over- and under-production are penalised.

Because the DA term is a constant, the LP objective is purely:

    Maximise: -Σ |P[t] - Forecast[t]| × ID_price[t] × dt/1000
            - (terminal + recovery + spill penalties)

The LP naturally follows the forecast.  It only deviates when water
constraints force it (reservoir full → must produce, reservoir low → recovery).
The deviation cost is weighted by the live intraday price, so the LP prefers
to deviate during cheap intraday hours.
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

TERMINAL_FREE_ZONE = 100.0   # mm free zone — no penalty
PW_ZONE2_WIDTH     = 100.0
PW_ZONE3_WIDTH     = 100.0
PW_SLOPE2          =   2.0   # EUR/mm
PW_SLOPE3          =   8.0   # EUR/mm
PW_SLOPE4          =  25.0   # EUR/mm


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
# LP model builder
# ---------------------------------------------------------------------------

def _build_model(N, forecast, id_price, inflow_b, inflow_h,
                 lb0, lh0, target_lb, target_lh):
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

    # Intraday imbalance: linearised |production - forecast|
    # dev_pos[t] - dev_neg[t] = P[t] - forecast[t]
    # cost = id_price[t] × (dev_pos[t] + dev_neg[t]) × dt / 1000
    m.dev_pos = Var(m.T, domain=NonNegativeReals)
    m.dev_neg = Var(m.T, domain=NonNegativeReals)
    m.imbalance_con = Constraint(m.T, rule=lambda m, t:
        m.dev_pos[t] - m.dev_neg[t] == m.P_M2[t] + m.P_M1[t] - forecast[t])

    # Objective — no direct revenue term; DA settlement is a constant
    # Use abs(id_price) so negative intraday prices don't make imbalance a reward
    # (which would make the LP unbounded).
    imbalance_cost = sum(
        abs(id_price[t]) * (m.dev_pos[t] + m.dev_neg[t]) * TIMESTEP_HOURS / 1000.0
        for t in m.T)
    penalties = (
        - PW_SLOPE2 * (m.pw_b2 + m.pw_h2)
        - PW_SLOPE3 * (m.pw_b3 + m.pw_h3)
        - PW_SLOPE4 * (m.pw_b4 + m.pw_h4)
    )
    m.obj = Objective(expr=-imbalance_cost + penalties, sense=maximize)
    return m


# ---------------------------------------------------------------------------
# Day dispatch
# ---------------------------------------------------------------------------

def dispatch_day(day_df, lb0, lh0, target_lb, target_lh,
                 solver=None, **_kwargs):
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    forecast = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]
    demand   = day_df['Consumption_kW'].tolist()
    da_price = day_df['Day_Ahead_Price_EUR_MWh'].tolist()
    id_price = day_df['Intra_Day_Price_EUR_MWh'].tolist()
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    model  = _build_model(N, forecast, id_price, inflow_b, inflow_h,
                          lb0, lh0, target_lb, target_lh)
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    if status not in ('optimal', 'feasible'):
        print(f"  WARNING: solver status '{status}' on {day_df.loc[0,'DateTime'].date()} — filling with zeros")
        zeros = [0.0] * N
        attach_common_results(
            day_df, zeros, zeros, [lb0] * N, [lh0] * N, zeros, zeros,
            demand, da_price, target_lb, target_lh, mode_name='WATER_VALUE',
        )
        day_df['Opt_DA_Trading_EUR']   = [
            (forecast[t] - demand[t]) * TIMESTEP_HOURS * da_price[t] / 1000.0
            for t in range(N)]
        day_df['Opt_ID_Imbalance_EUR'] = [
            -abs(forecast[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]
        day_df['Opt_Energy_Trading_EUR'] = [
            day_df['Opt_DA_Trading_EUR'].iloc[t] + day_df['Opt_ID_Imbalance_EUR'].iloc[t]
            for t in range(N)]
        day_df['Forecast_Drift_kW'] = [-f for f in forecast]
        return day_df

    opt_m2      = [value(model.P_M2[t])     for t in range(N)]
    opt_m1      = [value(model.P_M1[t])     for t in range(N)]
    opt_lb      = [value(model.LB[t])       for t in range(N)]
    opt_lh      = [value(model.LH[t])       for t in range(N)]
    opt_spill_b = [value(model.spill_b[t])  for t in range(N)]
    opt_spill_h = [value(model.spill_h[t])  for t in range(N)]

    # attach_common_results sets a generic Opt_Energy_Trading_EUR — we override it below
    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, da_price, target_lb, target_lh,
        mode_name='WATER_VALUE',
    )

    # Correct pricing
    da_component = [
        (forecast[t] - demand[t]) * TIMESTEP_HOURS * da_price[t] / 1000.0
        for t in range(N)]
    id_component = [
        -abs(opt_m2[t] + opt_m1[t] - forecast[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
        for t in range(N)]

    day_df['Opt_DA_Trading_EUR']   = da_component
    day_df['Opt_ID_Imbalance_EUR'] = id_component
    day_df['Opt_Energy_Trading_EUR'] = [da_component[t] + id_component[t] for t in range(N)]

    day_df['Forecast_Drift_kW'] = [
        (opt_m2[t] + opt_m1[t]) - forecast[t] for t in range(N)]

    return day_df
