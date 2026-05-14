"""
production_forecast_lp.py
=========================
Day-ahead production forecast — economically optimal hydro schedule.

What this is
------------
Called once per simulated day from run_optimiser.py when
PRODUCTION_FORECAST == 'LP_OPTIMISED'. Builds and solves a one-day LP that
mirrors the real operator's midnight decision:

    "Given yesterday's actual reservoir levels, expected inflows, forecast
     consumption, day-ahead prices, the running monthly peak so far, and
     the seasonal end-of-day level targets — what production schedule
     maximises tomorrow's expected revenue under all physical constraints?"

The returned 5-min P_M2[t] + P_M1[t] schedule replaces day_df['Forecast_kW']
for that day, becoming the DA bid AND the intraday dispatch target.

Inputs (all available at midnight in real operation)
----------------------------------------------------
  starting reservoir levels lb0, lh0  : yesterday's actual end state
  inflows  (perfect knowledge proxy)  : real measured inflows
  forecast consumption (CNN-LSTM)     : day_df['Forecast_Consumption_kW']
  DA prices                           : day_df['Day_Ahead_Price_EUR_MWh']
  BG_Long / BG_Short forecast         : Rolling-Mean variant (always)
  running monthly peak                : kW already paid for this month
  seasonal end-of-day target          : reservoir levels for asymmetric penalty

Constraints (all STRICT — physical reality)
-------------------------------------------
  Bidmi level     ∈ [1000, 2200] mm
  Haselholz level ∈ [600, 2800] mm
  M2 power        ∈ [0, 1920] kW
  M1 power        ∈ [0, 1156] kW
  Total ramp      |(M2+M1)[t] - (M2+M1)[t-RAMP_WINDOW]| ≤ 200 kW
  Water balance   Bidmi -> M2 -> Haselholz -> M1 (cascade)
  Spill variables absorb over-fill (relief valves)

Objective
---------
  max  DA_revenue
       - peak_penalty (PEAK_TARIFF × excess over running monthly peak,
                       only in [PEAK_TARIFF_HOUR_START, PEAK_TARIFF_HOUR_END))
       - terminal_penalty  (asymmetric: punitive below target, lenient above)

DA_revenue = sum_t (P_M2[t] + P_M1[t] - forecast_cons[t]) * DA_price[t] * dt / 1000

The BG forecast prices are accepted but not used in the LP objective: since the
LP itself chooses both the bid and the planned production, in-LP imbalance is
zero by construction. BG forecasts matter only for downstream intraday dispatch.

No battery: the DA bid is purely a hydro schedule. Battery is dispatched
intraday after this LP runs.
"""

import sys

from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals,
    Constraint, Objective, maximize, value, SolverFactory,
)
from hydro_constants import (
    P_MAX_M2, P_MAX_M1, RAMP_MAX, RAMP_WINDOW, TIMESTEP_HOURS,
    BIDMI_LEVEL_MIN, BIDMI_LEVEL_MAX,
    HASELHOLZ_LEVEL_MIN, HASELHOLZ_LEVEL_MAX,
    BIDMI_LS_PER_MM, HASELHOLZ_LS_PER_MM,
    COEFF_M2_BIDMI, COEFF_M2_CASCADE, COEFF_M1_HASELHOLZ,
    PEAK_TARIFF_EUR_KW,
    PEAK_TARIFF_HOUR_START, PEAK_TARIFF_HOUR_END,
)

# ---------------------------------------------------------------------------
# Terminal reservoir penalty parameters (asymmetric)
# ---------------------------------------------------------------------------
# BELOW target: punitive piecewise penalty (lost water value).
# ABOVE target: light flat penalty (slightly discourages overshoot, but allowed).
TERMINAL_FREE_ZONE_BELOW = 100.0    # mm — free zone immediately below target
PW_ZONE2_WIDTH           = 100.0    # mm — width of zone 2 (mild)
PW_ZONE3_WIDTH           = 100.0    # mm — width of zone 3 (moderate)
PW_SLOPE2_BELOW          =  0.05    # EUR/mm  (mild)
PW_SLOPE3_BELOW          =  0.15    # EUR/mm  (moderate)
PW_SLOPE4_BELOW          =  0.40    # EUR/mm  (punitive — depths below target are expensive)

TERMINAL_FREE_ZONE_ABOVE = 200.0    # mm — generous free zone above target
PW_SLOPE_ABOVE           =  0.005   # EUR/mm — very gentle penalty above target


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def get_solver():
    solver = SolverFactory('glpk')
    if not solver.available():
        print("ERROR: GLPK not found. Install via: conda install glpk")
        sys.exit(1)
    return solver


# ---------------------------------------------------------------------------
# Main entry point — call once per simulated day
# ---------------------------------------------------------------------------

def compute_optimal_forecast(day_df, lb0, lh0, target_lb, target_lh,
                              running_peak_kW, solver):
    """Solve the day-ahead LP and return the optimal 5-min production schedule.

    Parameters
    ----------
    day_df          : pandas DataFrame for one day (~288 rows). Required columns:
                       DateTime, Forecast_Consumption_kW, Day_Ahead_Price_EUR_MWh,
                       Bidmi_Inflow_ls, Haselholz_Inflow_ls.
                       (BG_Long_RollingMean_EUR_MWh / BG_Short_RollingMean_EUR_MWh
                        are accepted but currently unused in the objective.)
    lb0, lh0        : starting reservoir levels [mm] — yesterday's actual end
    target_lb,
    target_lh       : seasonal end-of-day targets [mm]
    running_peak_kW : monthly peak import already paid for this month [kW]
    solver          : Pyomo solver instance (from get_solver())

    Returns
    -------
    schedule : list of length N — optimal P_M2[t] + P_M1[t] [kW] for each 5-min step.
               On solver failure, returns [0.0] * N.
    """
    N     = len(day_df)
    T     = list(range(N))
    T_ext = list(range(N + 1))

    forecast_cons = day_df['Forecast_Consumption_kW'].tolist()
    # Use the forecast variant (may be degraded); settlement uses the real column.
    da_price      = day_df['Day_Ahead_Price_Forecast_EUR_MWh'].tolist()
    inflow_b      = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h      = day_df['Haselholz_Inflow_ls'].tolist()
    hours         = day_df['DateTime'].dt.hour.tolist()

    m = ConcreteModel()
    m.T     = Set(initialize=T)
    m.T_ext = Set(initialize=T_ext)

    # ── Decision variables: turbine production ─────────────────────────
    m.P_M2 = Var(m.T, domain=NonNegativeReals, bounds=(0.0, P_MAX_M2))
    m.P_M1 = Var(m.T, domain=NonNegativeReals, bounds=(0.0, P_MAX_M1))

    # ── Reservoir levels: hard bounds via Var bounds ───────────────────
    m.LB = Var(m.T_ext, bounds=(BIDMI_LEVEL_MIN,     BIDMI_LEVEL_MAX))
    m.LH = Var(m.T_ext, bounds=(HASELHOLZ_LEVEL_MIN, HASELHOLZ_LEVEL_MAX))
    m.LB[0].fix(lb0)
    m.LH[0].fix(lh0)

    # ── Spill variables (relief valves for over-fill) ──────────────────
    m.spill_b = Var(m.T, domain=NonNegativeReals)
    m.spill_h = Var(m.T, domain=NonNegativeReals)

    # ── Water balance (cascade Bidmi -> M2 -> Haselholz -> M1) ─────────
    m.wb_b = Constraint(m.T, rule=lambda m, t:
        m.LB[t + 1] == m.LB[t] + inflow_b[t] / BIDMI_LS_PER_MM
                     - COEFF_M2_BIDMI * m.P_M2[t] - m.spill_b[t])
    m.wb_h = Constraint(m.T, rule=lambda m, t:
        m.LH[t + 1] == m.LH[t] + inflow_h[t] / HASELHOLZ_LS_PER_MM
                     + COEFF_M2_CASCADE  * m.P_M2[t]
                     - COEFF_M1_HASELHOLZ * m.P_M1[t] - m.spill_h[t])

    # ── Total ramp constraint: |(M2+M1)[t] - (M2+M1)[t-W]| ≤ RAMP_MAX ──
    def _ramp_total(m, t, direction):
        if t < RAMP_WINDOW:
            return Constraint.Skip
        now  = m.P_M2[t]              + m.P_M1[t]
        prev = m.P_M2[t - RAMP_WINDOW] + m.P_M1[t - RAMP_WINDOW]
        diff = now - prev if direction == 'up' else prev - now
        return diff <= RAMP_MAX

    m.ramp_up = Constraint(m.T, rule=lambda m, t: _ramp_total(m, t, 'up'))
    m.ramp_dn = Constraint(m.T, rule=lambda m, t: _ramp_total(m, t, 'dn'))

    # ── DA revenue ─────────────────────────────────────────────────────
    # bid[t] = P_M2[t] + P_M1[t] - forecast_consumption[t]
    # (positive = selling on DA, negative = buying)
    da_revenue = sum(
        (m.P_M2[t] + m.P_M1[t] - forecast_cons[t])
            * da_price[t] * TIMESTEP_HOURS / 1000.0
        for t in m.T)

    # ── Peak fee penalty (Leistungsentgelt) ────────────────────────────
    # P_import[t] = max(0, forecast_cons[t] - P_M2[t] - P_M1[t])  (no battery)
    # excess_peak >= avg(P_import in tariff-window blocks) - running_peak_kW
    B_BLOCKS = N // 3

    m.P_import = Var(m.T, domain=NonNegativeReals)
    m.import_def = Constraint(m.T, rule=lambda m, t:
        m.P_import[t] >= forecast_cons[t] - (m.P_M2[t] + m.P_M1[t]))

    m.excess_peak = Var(domain=NonNegativeReals)
    tariff_blocks = [
        b for b in range(B_BLOCKS)
        if all(PEAK_TARIFF_HOUR_START <= hours[3 * b + i] < PEAK_TARIFF_HOUR_END
               for i in range(3))
    ]
    if tariff_blocks:
        m.peak_block = Constraint(
            tariff_blocks,
            rule=lambda m, b: m.excess_peak >= (
                m.P_import[3 * b] + m.P_import[3 * b + 1] + m.P_import[3 * b + 2]
            ) / 3.0 - running_peak_kW,
        )
    peak_penalty = PEAK_TARIFF_EUR_KW * m.excess_peak

    # ── Asymmetric terminal reservoir penalty ──────────────────────────
    # BELOW target (lb_end < target): punitive piecewise (zones 2/3/4)
    # ABOVE target (lb_end > target): light flat penalty after free zone
    m.dev_b_below = Var(domain=NonNegativeReals)
    m.dev_b_above = Var(domain=NonNegativeReals)
    m.dev_h_below = Var(domain=NonNegativeReals)
    m.dev_h_above = Var(domain=NonNegativeReals)
    m.dev_b_below_con = Constraint(expr=m.dev_b_below >= target_lb - m.LB[N])
    m.dev_b_above_con = Constraint(expr=m.dev_b_above >= m.LB[N] - target_lb)
    m.dev_h_below_con = Constraint(expr=m.dev_h_below >= target_lh - m.LH[N])
    m.dev_h_above_con = Constraint(expr=m.dev_h_above >= m.LH[N] - target_lh)

    # Below: zone 2 (mild) -> zone 3 (moderate) -> zone 4 (punitive)
    m.pw_b_b2 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE2_WIDTH))
    m.pw_b_b3 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE3_WIDTH))
    m.pw_b_b4 = Var(domain=NonNegativeReals)
    m.pw_b_b_link = Constraint(expr=
        m.pw_b_b2 + m.pw_b_b3 + m.pw_b_b4
        >= m.dev_b_below - TERMINAL_FREE_ZONE_BELOW)

    m.pw_h_b2 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE2_WIDTH))
    m.pw_h_b3 = Var(domain=NonNegativeReals, bounds=(0, PW_ZONE3_WIDTH))
    m.pw_h_b4 = Var(domain=NonNegativeReals)
    m.pw_h_b_link = Constraint(expr=
        m.pw_h_b2 + m.pw_h_b3 + m.pw_h_b4
        >= m.dev_h_below - TERMINAL_FREE_ZONE_BELOW)

    # Above: single flat penalty after a generous free zone
    m.pw_b_a = Var(domain=NonNegativeReals)
    m.pw_h_a = Var(domain=NonNegativeReals)
    m.pw_b_a_link = Constraint(expr=
        m.pw_b_a >= m.dev_b_above - TERMINAL_FREE_ZONE_ABOVE)
    m.pw_h_a_link = Constraint(expr=
        m.pw_h_a >= m.dev_h_above - TERMINAL_FREE_ZONE_ABOVE)

    terminal_penalty = (
          PW_SLOPE2_BELOW * (m.pw_b_b2 + m.pw_h_b2)
        + PW_SLOPE3_BELOW * (m.pw_b_b3 + m.pw_h_b3)
        + PW_SLOPE4_BELOW * (m.pw_b_b4 + m.pw_h_b4)
        + PW_SLOPE_ABOVE  * (m.pw_b_a  + m.pw_h_a)
    )

    # ── Objective ──────────────────────────────────────────────────────
    m.obj = Objective(expr=da_revenue - peak_penalty - terminal_penalty,
                       sense=maximize)

    # ── Solve ──────────────────────────────────────────────────────────
    result = solver.solve(m, tee=False)
    status = str(result.solver.termination_condition)
    if status not in ('optimal', 'feasible'):
        date_str = day_df['DateTime'].iloc[0].date()
        print(f"  PRODUCTION FORECAST LP failed on {date_str}: status='{status}'")
        return [0.0] * N

    return [value(m.P_M2[t]) + value(m.P_M1[t]) for t in T]