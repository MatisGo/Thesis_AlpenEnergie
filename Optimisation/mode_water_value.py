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
    TURBINE_CASCADE_RATIO, PEAK_TARIFF_EUR_KW,
    water_balance_step, spill_to_kwh, attach_common_results,
)
from battery_control import (
    EFF_CHARGE, EFF_DISCHARGE, SOC_MIN, SOC_MAX, _P_MAX_BATT,
    SOC_TARGET_LO_PCT, SOC_TARGET_HI_PCT, SOC_PENALTY_EUR_KWH,
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
                 lb0, lh0, target_lb, target_lh, battery_cfg=None,
                 running_peak_kW=0.0, demand=None):
    """
    Build the WATER_VALUE LP.

    battery_cfg : BatteryConfig or None.
      When mode == 'INTRADAY', battery variables (charge, discharge, SOC) are
      added to the LP.  The imbalance constraint is written on the NET GRID
      export (turbine + battery_net) so the LP can use the battery to reduce
      ID costs.  The LP stays fully linear (no binary variables needed).
    """
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

    # Water balance — uses turbine output only (battery is at grid connection,
    # not in the water circuit)
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

    # Intraday imbalance vars — linearised |grid_export - forecast|
    m.dev_pos = Var(m.T, domain=NonNegativeReals)
    m.dev_neg = Var(m.T, domain=NonNegativeReals)

    # Battery variables (INTRADAY battery mode)
    with_battery = battery_cfg is not None and battery_cfg.mode == 'INTRADAY'
    if with_battery:
        soc0_clamped = max(SOC_MIN, min(SOC_MAX, battery_cfg.soc0))
        m.batt_c   = Var(m.T, domain=NonNegativeReals, bounds=(0, _P_MAX_BATT))
        m.batt_d   = Var(m.T, domain=NonNegativeReals, bounds=(0, _P_MAX_BATT))
        m.batt_SOC = Var(m.T_ext, domain=NonNegativeReals)
        m.batt_SOC[0].fix(soc0_clamped)
        m.batt_soc_lo = Constraint([t for t in T_ext if t >= 1],
                                   rule=lambda m, t: m.batt_SOC[t] >= SOC_MIN)
        m.batt_soc_hi = Constraint([t for t in T_ext if t >= 1],
                                   rule=lambda m, t: m.batt_SOC[t] <= SOC_MAX)
        m.batt_soc_bal = Constraint(m.T, rule=lambda m, t:
            m.batt_SOC[t + 1] == m.batt_SOC[t]
                               + m.batt_c[t] * EFF_CHARGE    * TIMESTEP_HOURS
                               - m.batt_d[t] / EFF_DISCHARGE * TIMESTEP_HOURS)
        # Imbalance on NET GRID export: turbine + battery_net
        m.imbalance_con = Constraint(m.T, rule=lambda m, t:
            m.dev_pos[t] - m.dev_neg[t] ==
                m.P_M2[t] + m.P_M1[t] + m.batt_d[t] - m.batt_c[t] - forecast[t])
    else:
        # No battery — imbalance on turbine output only
        m.imbalance_con = Constraint(m.T, rule=lambda m, t:
            m.dev_pos[t] - m.dev_neg[t] == m.P_M2[t] + m.P_M1[t] - forecast[t])

    # -----------------------------------------------------------------------
    # Grid import peak tariff (Leistungsentgelt)
    # P_import[t] = max(0, demand[t] - local_generation[t])
    # 15-min blocks: 3 consecutive 5-min steps.
    # excess_peak >= avg(P_import in block b) - running_peak_kW  ∀ b
    # Objective penalty: PEAK_TARIFF_EUR_KW × excess_peak
    # -----------------------------------------------------------------------
    load = demand if demand is not None else [0.0] * N
    B_BLOCKS = N // 3

    m.P_import = Var(m.T, domain=NonNegativeReals)
    if with_battery:
        m.import_def = Constraint(m.T, rule=lambda m, t:
            m.P_import[t] >= load[t] - (m.P_M2[t] + m.P_M1[t]) - (m.batt_d[t] - m.batt_c[t]))
    else:
        m.import_def = Constraint(m.T, rule=lambda m, t:
            m.P_import[t] >= load[t] - (m.P_M2[t] + m.P_M1[t]))

    m.excess_peak = Var(domain=NonNegativeReals)
    if B_BLOCKS > 0:
        m.peak_block = Constraint(
            list(range(B_BLOCKS)),
            rule=lambda m, b: m.excess_peak >= (
                m.P_import[3 * b] + m.P_import[3 * b + 1] + m.P_import[3 * b + 2]
            ) / 3.0 - running_peak_kW,
        )

    # Objective — minimise ID imbalance cost + terminal penalties + peak tariff
    imbalance_cost = sum(
        abs(id_price[t]) * (m.dev_pos[t] + m.dev_neg[t]) * TIMESTEP_HOURS / 1000.0
        for t in m.T)
    penalties = (
        - PW_SLOPE2 * (m.pw_b2 + m.pw_h2)
        - PW_SLOPE3 * (m.pw_b3 + m.pw_h3)
        - PW_SLOPE4 * (m.pw_b4 + m.pw_h4)
    )
    peak_penalty = PEAK_TARIFF_EUR_KW * m.excess_peak

    # End-of-day SOC flexibility reserve penalty
    soc_penalty = 0
    if with_battery:
        usable        = SOC_MAX - SOC_MIN
        soc_target_lo = SOC_MIN + SOC_TARGET_LO_PCT * usable
        soc_target_hi = SOC_MIN + SOC_TARGET_HI_PCT * usable
        m.dev_soc_lo = Var(domain=NonNegativeReals)
        m.dev_soc_hi = Var(domain=NonNegativeReals)
        m.soc_dev_lo_con = Constraint(expr=m.dev_soc_lo >= soc_target_lo - m.batt_SOC[N])
        m.soc_dev_hi_con = Constraint(expr=m.dev_soc_hi >= m.batt_SOC[N] - soc_target_hi)
        soc_penalty = SOC_PENALTY_EUR_KWH * (m.dev_soc_lo + m.dev_soc_hi)

    m.obj = Objective(expr=-imbalance_cost + penalties - peak_penalty - soc_penalty, sense=maximize)
    return m


# ---------------------------------------------------------------------------
# Day dispatch
# ---------------------------------------------------------------------------

def dispatch_day(day_df, lb0, lh0, target_lb, target_lh,
                 solver=None, battery_cfg=None, running_peak_kW=0.0, **_kwargs):
    """
    battery_cfg    : BatteryConfig or None.
      'INTRADAY' mode → battery vars added to LP; LP jointly minimises
                        ID imbalance on net grid export (hydro + battery).
                        Battery columns and _batt_soc_end written to day_df.
      'DAY_AHEAD' mode or None → battery_cfg ignored here; handled separately.
    running_peak_kW: current monthly peak import already observed [kW].
                     The LP will only be penalised for exceeding this value.
    """
    day_df      = day_df.reset_index(drop=True)
    N           = len(day_df)
    forecast    = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]
    demand      = day_df['Consumption_kW'].tolist()
    da_price    = day_df['Day_Ahead_Price_EUR_MWh'].tolist()
    id_price    = day_df['Intra_Day_Price_EUR_MWh'].tolist()
    inflow_b    = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h    = day_df['Haselholz_Inflow_ls'].tolist()
    with_batt   = battery_cfg is not None and battery_cfg.mode == 'INTRADAY'

    model  = _build_model(N, forecast, id_price, inflow_b, inflow_h,
                          lb0, lh0, target_lb, target_lh,
                          battery_cfg=battery_cfg if with_batt else None,
                          running_peak_kW=running_peak_kW, demand=demand)
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    if status not in ('optimal', 'feasible'):
        print(f"  WARNING: solver status '{status}' on {day_df.loc[0,'DateTime'].date()} — filling with zeros")
        day_df._failed = 1
        zeros = [0.0] * N
        attach_common_results(
            day_df, zeros, zeros, [lb0] * N, [lh0] * N, zeros, zeros,
            demand, target_lb, target_lh, mode_name='WATER_VALUE',
        )
        day_df['Opt_DA_Trading_EUR']     = [
            (forecast[t] - demand[t]) * TIMESTEP_HOURS * da_price[t] / 1000.0
            for t in range(N)]
        day_df['Opt_ID_Imbalance_EUR']   = [
            -abs(forecast[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]
        day_df['Opt_Energy_Trading_EUR'] = [
            day_df['Opt_DA_Trading_EUR'].iloc[t] + day_df['Opt_ID_Imbalance_EUR'].iloc[t]
            for t in range(N)]
        day_df['Forecast_Drift_kW'] = [-f for f in forecast]
        if with_batt:
            day_df['Batt_Charge_kW']    = zeros
            day_df['Batt_Discharge_kW'] = zeros
            day_df['Batt_SOC_kWh']      = [battery_cfg.soc0] * N
            day_df['Batt_Net_kW']       = zeros
            day_df['Batt_Revenue_EUR']  = zeros
            day_df._batt_soc_end        = battery_cfg.soc0
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
        demand, target_lb, target_lh,
        mode_name='WATER_VALUE',
    )

    da_component = [
        (forecast[t] - demand[t]) * TIMESTEP_HOURS * da_price[t] / 1000.0
        for t in range(N)]

    if with_batt:
        batt_c   = [value(model.batt_c[t])   for t in range(N)]
        batt_d   = [value(model.batt_d[t])   for t in range(N)]
        batt_soc = [value(model.batt_SOC[t]) for t in range(N)]

        grid_out = [opt_m2[t] + opt_m1[t] + batt_d[t] - batt_c[t] for t in range(N)]
        id_component = [
            -abs(grid_out[t] - forecast[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]
        id_hydro_only = [
            -abs(opt_m2[t] + opt_m1[t] - forecast[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]

        day_df['Batt_Charge_kW']    = batt_c
        day_df['Batt_Discharge_kW'] = batt_d
        day_df['Batt_SOC_kWh']      = batt_soc
        day_df['Batt_Net_kW']       = [batt_d[t] - batt_c[t] for t in range(N)]
        day_df['Batt_Revenue_EUR']  = [id_component[t] - id_hydro_only[t] for t in range(N)]
        day_df._batt_soc_end        = value(model.batt_SOC[N])
    else:
        id_component = [
            -abs(opt_m2[t] + opt_m1[t] - forecast[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]

    day_df['Opt_DA_Trading_EUR']     = da_component
    day_df['Opt_ID_Imbalance_EUR']   = id_component
    day_df['Opt_Energy_Trading_EUR'] = [da_component[t] + id_component[t] for t in range(N)]
    day_df['Forecast_Drift_kW']      = [
        (opt_m2[t] + opt_m1[t]) - forecast[t] for t in range(N)]

    return day_df
