"""
battery_control.py
==================
Battery energy storage system (BESS) dispatch modes.

Battery specifications
----------------------
  CAPACITY_KWH     : 6 000 kWh  — total usable storage (6 MWh)
  EFF_CHARGE       : √0.90     — charge efficiency  (grid  → battery, ≈ 0.9487)
  EFF_DISCHARGE    : √0.90     — discharge efficiency (battery → grid, ≈ 0.9487)
  ROUND_TRIP_EFF   : 0.90      — EFF_CHARGE × EFF_DISCHARGE
  SOC_MIN_PCT      : 0.10      — lower SOC limit (10 % of capacity = 0.6 kWh)
  SOC_MAX_PCT      : 0.90      — upper SOC limit (90 % of capacity = 5.4 kWh)
  SOC_INITIAL_PCT  : 0.10      — SOC at start of first day (10 % = 0.6 kWh)
  CYCLE_KWH        : 6 000 kWh  — energy discharged that counts as one full cycle (6 MWh)
  C_RATE           : 0.5 C      — inverter/hardware power limit (50 % of capacity per hour)
  P_MAX_BATT       : C_RATE × CAPACITY_KWH  [kW]  — applies to LP bounds and rule-based controller

Modes
-----
  DAY_AHEAD : LP that buys at low DA prices and sells at high DA prices.
              Prices for the full day are known before dispatch → globally optimal.
              The battery is independent of the hydro plant.

  INTRADAY  : Rule-based causal controller triggered by reservoir fill levels.
              Thresholds (90 % / 35 % of usable range) are checked each timestep
              using the reservoir levels produced by the hydro dispatch.

              Charge when either reservoir fill > 90 %:
                Power scales from 0 (at threshold) to P_MAX_BATT (reservoir full),
                capped by remaining SOC headroom.  Turbine produces Forecast + charge_kw,
                grid sees Forecast.  Water is released faster (prevents spill).

              Discharge when either reservoir fill < 30 %:
                Power scales from 0 (at threshold) to P_MAX_BATT (reservoir empty),
                capped by available SOC above SOC_MIN.  Turbine produces Forecast − discharge_kw,
                battery covers the gap, grid sees Forecast.  Water is saved (avoids running dry).

              Opt_Production_kW is updated to reflect the adjusted turbine output.
              Opt_Energy_Trading_EUR is recalculated on the actual grid output.
              Batt_Revenue_EUR captures the ID savings vs hydro-only operation.
              No future information is used; the controller is fully causal.
              Can be deactivated by setting BATTERY_ACTIVE = False in run_optimiser.py.

Output columns added to day_df
-------------------------------
  Batt_Charge_kW    power drawn from the grid     [kW]
  Batt_Discharge_kW power injected to the grid    [kW]
  Batt_SOC_kWh      state of charge (start of t)  [kWh]
  Batt_Net_kW       discharge − charge             [kW]  (positive = exporting)
  Batt_Revenue_EUR  (discharge − charge) × price × dt / 1000  [EUR]
"""

import math
import sys
from dataclasses import dataclass

from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals,
    Constraint, Objective, maximize, value, SolverFactory,
)
from hydro_constants import TIMESTEP_HOURS


# ---------------------------------------------------------------------------
# Configuration object  (passed from run_optimiser → hydro dispatch functions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatteryConfig:
    """Immutable battery configuration passed into every dispatch_day function.

    mode             : 'DAY_AHEAD' — LP arbitrage, fully independent of hydro.
                       'INTRADAY'  — reservoir-triggered co-simulation inside hydro dispatch.
                       'HYBRID'    — INTRADAY co-sim with optional DA block reservation.
    soc0             : SOC at start of the day [kWh], carried from previous day.
    soc_max_override : When set, replaces SOC_MAX for the INTRADAY controller.
                       Used by HYBRID mode to cap the INTRADAY zone below the DA
                       block floor, so committed energy stays reserved.
                       None → use the global SOC_MAX.

    Pass None instead of a BatteryConfig to signal battery inactive.
    """
    mode: str
    soc0: float
    soc_max_override: float = None

# ---------------------------------------------------------------------------
# Battery specifications
# ---------------------------------------------------------------------------

CAPACITY_KWH    = 200   # 1 MWh
EFF_CHARGE      = math.sqrt(0.80)          
EFF_DISCHARGE   = math.sqrt(0.80)          
ROUND_TRIP_EFF  = EFF_CHARGE * EFF_DISCHARGE  # = 0.80

SOC_MIN_PCT     = 0.10
SOC_MAX_PCT     = 0.90
SOC_INITIAL_PCT = 0.10

# Round to avoid floating-point residuals (e.g. 0.10 × 6000 = 600.0000000001)
# that cause GLPK constraint micro-violations and 'other' solver status.
SOC_MIN     = round(SOC_MIN_PCT     * CAPACITY_KWH, 6)   #   600.0 kWh
SOC_MAX     = round(SOC_MAX_PCT     * CAPACITY_KWH, 6)   # 5 400.0 kWh
SOC_INITIAL = round(SOC_INITIAL_PCT * CAPACITY_KWH, 6)   #   600.0 kWh

CYCLE_KWH   = CAPACITY_KWH                               # 1 full cycle = full usable capacity

# ---------------------------------------------------------------------------
# C-rate — inverter / hardware power limit
# ---------------------------------------------------------------------------
# 0.5 C means the battery can charge or discharge at 50 % of its nominal
# capacity per hour.  For a 200 kWh battery: P_max = 0.5 × 200 = 100 kW.
# This value caps both the rule-based controller and the LP variable bounds.
C_RATE      = 1                               # C  (change here to adjust power limit)
_P_MAX_BATT = round(C_RATE * CAPACITY_KWH, 3)  # kW — used as LP upper bound


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def get_battery_solver():
    solver = SolverFactory('glpk')
    if not solver.available():
        print("ERROR: GLPK not found.  Install via: conda install glpk")
        sys.exit(1)
    return solver


# ---------------------------------------------------------------------------
# DAY_AHEAD LP
# ---------------------------------------------------------------------------

def _build_da_model(N, da_price, soc0):
    """
    LP: maximise day-ahead arbitrage revenue.

    SOC balance (per timestep dt = TIMESTEP_HOURS):
      SOC[t+1] = SOC[t]
               + charge[t]    × EFF_CHARGE    × dt   (energy stored)
               - discharge[t] / EFF_DISCHARGE × dt   (energy drawn from battery)

    charge[t] and discharge[t] are never simultaneously positive at optimum
    because round-trip efficiency < 1 makes simultaneous use unprofitable.
    No explicit mutual-exclusion constraint needed.
    """
    m     = ConcreteModel()
    T     = list(range(N))
    T_ext = list(range(N + 1))
    m.T     = Set(initialize=T)
    m.T_ext = Set(initialize=T_ext)

    # Decision variables — bounded by max achievable power in one timestep
    # (tight physical ceiling; keeps GLPK numerically stable)
    m.charge    = Var(m.T, domain=NonNegativeReals, bounds=(0, _P_MAX_BATT))
    m.discharge = Var(m.T, domain=NonNegativeReals, bounds=(0, _P_MAX_BATT))

    # State of charge
    # SOC[0] is fixed — apply bounds only to t >= 1 to avoid floating-point
    # conflicts between fix() and bounds (causes GLPK 'other'/infeasible status).
    m.SOC = Var(m.T_ext, domain=NonNegativeReals)
    m.SOC[0].fix(max(SOC_MIN, min(SOC_MAX, soc0)))
    m.soc_lo = Constraint([t for t in T_ext if t >= 1],
                          rule=lambda m, t: m.SOC[t] >= SOC_MIN)
    m.soc_hi = Constraint([t for t in T_ext if t >= 1],
                          rule=lambda m, t: m.SOC[t] <= SOC_MAX)

    # SOC balance
    m.soc_bal = Constraint(m.T, rule=lambda m, t:
        m.SOC[t + 1] == m.SOC[t]
                       + m.charge[t]    * EFF_CHARGE    * TIMESTEP_HOURS
                       - m.discharge[t] / EFF_DISCHARGE * TIMESTEP_HOURS)

    # Objective: maximise DA revenue
    m.obj = Objective(
        expr=sum(
            (m.discharge[t] - m.charge[t]) * da_price[t] * TIMESTEP_HOURS / 1000.0
            for t in m.T),
        sense=maximize,
    )
    return m


# ---------------------------------------------------------------------------
# INTRADAY mode — per-step causal controller  (called inside hydro dispatch)
# ---------------------------------------------------------------------------

# Reservoir fill thresholds that trigger battery action
CHARGE_THRESHOLD    = 0.90   # charge when fill > 90 %
DISCHARGE_THRESHOLD = 0.30   # discharge when fill < 30 %
# Power is NOT fixed — it scales proportionally with how far the fill is from
# the threshold, up to _P_MAX_BATT and capped by available SOC headroom.


def battery_step_forecast(soc, fill_b, fill_h, soc_max_eff=None):
    """
    Compute one 5-min timestep of the INTRADAY / HYBRID battery controller.

    Called at every timestep from inside the hydro dispatch loop so that
    the battery decision and the water-balance update share the same clock.
    No future information is used — only the current SOC and fill levels.

    Power is proportional to the reservoir deviation from its threshold:

      charge_urgency   = max(fill_b, fill_h) - CHARGE_THRESHOLD
                         normalised to [0, 1] over [threshold → 1.0]
      discharge_urgency = DISCHARGE_THRESHOLD - min(fill_b, fill_h)
                          normalised to [0, 1] over [threshold → 0.0]

      actual_power = urgency × _P_MAX_BATT, capped by available SOC headroom.

    Priority: charging (high reservoir) beats discharging (low reservoir).

    Parameters
    ----------
    soc         : current state of charge [kWh]  (start of timestep)
    fill_b      : Bidmi fill ratio     [0..1]
    fill_h      : Haselholz fill ratio [0..1]
    soc_max_eff : effective SOC upper limit [kWh]; if None, uses global SOC_MAX.
                  Set by HYBRID mode to DA_BLOCK_FLOOR_SOC so the INTRADAY
                  controller cannot consume reserved DA energy.

    Returns
    -------
    charge_kw    : power drawn into battery this step  [kW]
    discharge_kw : power pushed out of battery         [kW]
    soc_new      : SOC at end of timestep              [kWh]
    """
    _soc_max = soc_max_eff if soc_max_eff is not None else SOC_MAX

    # --- Charge urgency: worst (highest) reservoir above threshold ---
    fill_excess = max(fill_b - CHARGE_THRESHOLD, fill_h - CHARGE_THRESHOLD, 0.0)
    charge_urgency = min(fill_excess / (1.0 - CHARGE_THRESHOLD), 1.0)

    # --- Discharge urgency: worst (lowest) reservoir below threshold ---
    fill_deficit = max(DISCHARGE_THRESHOLD - fill_b, DISCHARGE_THRESHOLD - fill_h, 0.0)
    discharge_urgency = min(fill_deficit / DISCHARGE_THRESHOLD, 1.0)

    if charge_urgency > 0.0:
        # Charge: proportional power, capped by SOC headroom
        max_c = max(0.0, (_soc_max - soc) / (EFF_CHARGE * TIMESTEP_HOURS))
        c = min(_P_MAX_BATT * charge_urgency, max_c)
        d = 0.0
    elif discharge_urgency > 0.0:
        # Discharge: proportional power, capped by available SOC
        max_d = max(0.0, (soc - SOC_MIN) * EFF_DISCHARGE / TIMESTEP_HOURS)
        d = min(_P_MAX_BATT * discharge_urgency, max_d)
        c = 0.0
    else:
        c = d = 0.0

    soc_new = soc + c * EFF_CHARGE * TIMESTEP_HOURS - d / EFF_DISCHARGE * TIMESTEP_HOURS
    soc_new = max(SOC_MIN, min(_soc_max, soc_new))
    return c, d, soc_new


# ---------------------------------------------------------------------------
# Day dispatch
# ---------------------------------------------------------------------------

def dispatch_day_battery(day_df, soc0, solver, mode='DAY_AHEAD'):
    """
    Run battery dispatch for one day.

    Parameters
    ----------
    day_df  : DataFrame for the day (must contain Day_Ahead_Price_EUR_MWh,
              Intra_Day_Price_EUR_MWh, Opt_Production_kW, Forecast_kW)
    soc0    : starting SOC [kWh]
    solver  : Pyomo solver (from get_battery_solver()); unused for FORECAST mode
    mode    : 'DAY_AHEAD' | 'INTRADAY'

    Returns
    -------
    day_df          : same DataFrame with battery columns added
    soc_end         : SOC at end of day [kWh]  (carry-over to next day)
    discharged_kwh  : total energy discharged today [kWh]  (for cycle counting)
    """
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    da_price = day_df['Day_Ahead_Price_EUR_MWh'].tolist()

    if mode != 'DAY_AHEAD':
        raise ValueError(
            f"dispatch_day_battery only handles 'DAY_AHEAD'. "
            f"INTRADAY battery is co-simulated inside the hydro dispatch loop.")

    model  = _build_da_model(N, da_price, soc0)
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    if status not in ('optimal', 'feasible'):
        print(f"  BATTERY WARNING: solver '{status}' on "
              f"{day_df.loc[0, 'DateTime'].date()} — battery inactive")
        charge    = [0.0] * N
        discharge = [0.0] * N
        soc       = [soc0] * N
        soc_end   = soc0
    else:
        charge    = [value(model.charge[t])    for t in range(N)]
        discharge = [value(model.discharge[t]) for t in range(N)]
        soc       = [value(model.SOC[t])       for t in range(N)]
        soc_end   = value(model.SOC[N])

    day_df['Batt_Charge_kW']    = charge
    day_df['Batt_Discharge_kW'] = discharge
    day_df['Batt_SOC_kWh']      = soc
    day_df['Batt_Net_kW']       = [discharge[t] - charge[t] for t in range(N)]
    day_df['Batt_Revenue_EUR']  = [
        (discharge[t] - charge[t]) * da_price[t] * TIMESTEP_HOURS / 1000.0
        for t in range(N)]

    discharged_kwh = sum(discharge[t] * TIMESTEP_HOURS for t in range(N))

    return day_df, soc_end, discharged_kwh
