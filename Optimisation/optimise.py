"""
Hydro scheduling LP optimiser
==============================

For each calendar day in results.csv, solves an LP to find the optimal
combined production schedule that minimises total energy cost (net grid
purchases at spot price).

Day chaining
------------
The optimised end-of-day reservoir level becomes the starting level for
the following day. Only the very first day uses the actual observed level.
This makes decisions physically consistent across days: water used today
is unavailable tomorrow.

Seasonal target
---------------
The terminal constraint no longer says "return to where you started today."
Instead it penalises deviations from a smooth seasonal target curve derived
from a 30-day rolling average of actual midnight (00:00) reservoir levels.
This gives a physically meaningful reference that:
  - captures annual filling/draining cycles (spring snowmelt, winter draw-down)
  - provides a fixed restoring force that prevents multi-week drift
  - caps Haselholz targets at the physical upper bound (2800 mm)

Piecewise-linear penalty (LP-compatible)
-----------------------------------------
  |deviation from target|   CHF/mm
  0  – 100 mm              0   (free zone)
  100– 200 mm              PW_SLOPE2
  200– 300 mm              PW_SLOPE3
  > 300 mm                 PW_SLOPE4

The increasing slopes approximate an exponential effect while keeping
the problem a pure LP solvable by GLPK in milliseconds.

Mathematical formulation
------------------------
Sets:
  T     = {0,...,N-1}   production timesteps (288 @ 5-min resolution)
  T_ext = {0,...,N}     reservoir level timestep boundaries

Decision variables:
  P[t]  in [0, P_MAX]   combined production (kW)

State variables:
  LB[t] >= BIDMI_LEVEL_MIN      Bidmi reservoir level (mm)
  LH[t] >= HASELHOLZ_LEVEL_MIN  Haselholz reservoir level (mm)

Spill / slack:
  spill_b[t], spill_h[t] >= 0   overflow relief per timestep
  abs_dev_b, abs_dev_h   >= 0   absolute deviation from seasonal target
  pw_b2/3/4, pw_h2/3/4   >= 0   piecewise penalty zone variables

Objective:
  Minimize  sum_t [(Demand[t]-P[t]) * TIMESTEP_H * Price[t] / 1000]
          + PW_SLOPE2*(pw_b2+pw_h2) + PW_SLOPE3*(pw_b3+pw_h3)
          + PW_SLOPE4*(pw_b4+pw_h4)
          + SPILL_PENALTY * sum(spill_b + spill_h)

Usage:
  python optimise.py               # all days in results.csv
  python optimise.py 2025-06-15   # single day
"""

import os
import sys

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals,
    Constraint, Objective, minimize, value, SolverFactory,
)

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS  (must match run_model.py)
# ---------------------------------------------------------------------------

TURBINE_RATIO        = 1.560
BIDMI_KWH_PER_MM     = 3.33
HASELHOLZ_KWH_PER_MM = 0.455
TIMESTEP_HOURS       = 5 / 60

_DENOM  = 1.0 + TURBINE_RATIO
FRAC_B  = 1.0 / _DENOM
FRAC_H  = TURBINE_RATIO / _DENOM

COEFF_B = FRAC_B * TIMESTEP_HOURS / BIDMI_KWH_PER_MM
COEFF_H = FRAC_H * TIMESTEP_HOURS / HASELHOLZ_KWH_PER_MM

P_MAX_COMBINED    = 2856.0
P_MAX_BIDMI       = 1700.0
P_MAX_HASELHOLZ   = 1156.0

RAMP_MAX = 500.0

BIDMI_LEVEL_MIN     = 1000.0
BIDMI_LEVEL_MAX     = 2200.0
HASELHOLZ_LEVEL_MIN =  600.0
HASELHOLZ_LEVEL_MAX = 2800.0

# ---------------------------------------------------------------------------
# SEASONAL TARGET PARAMETERS
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 30   # days for smoothing midnight levels into seasonal curve

# ---------------------------------------------------------------------------
# PIECEWISE-LINEAR TERMINAL PENALTY
# ---------------------------------------------------------------------------
#
#  deviation from seasonal target   penalty rate
#  0   – 100 mm                     0 CHF/mm   (free zone)
#  100 – 200 mm                     PW_SLOPE2
#  200 – 300 mm                     PW_SLOPE3
#  > 300 mm                         PW_SLOPE4
#
# Cumulative cost at boundary:
#   200 mm :  2 * 100            =   200 CHF
#   300 mm :  200 + 8 * 100      =  1000 CHF
#   400 mm :  1000 + 25 * 100    =  3500 CHF
#
TERMINAL_FREE_ZONE = 100.0   # mm — no penalty within this band
PW_ZONE2_WIDTH     = 100.0   # mm width of zone 2
PW_ZONE3_WIDTH     = 100.0   # mm width of zone 3
PW_SLOPE2          =   2.0   # CHF/mm
PW_SLOPE3          =   8.0   # CHF/mm
PW_SLOPE4          =  25.0   # CHF/mm

# Spill penalty (CHF/mm) — must exceed max energy revenue per mm
# Bidmi: 3.33 kWh/mm * ~100 CHF/MWh / 1000 = 0.333 CHF/mm  ->  1.0 is 3x above
SPILL_PENALTY = 1.0

RESULTS_FILENAME = 'results.csv'
OUTPUT_FILENAME  = 'optimised_results.csv'


# ---------------------------------------------------------------------------
# SEASONAL TARGET COMPUTATION
# ---------------------------------------------------------------------------

def compute_seasonal_targets(df):
    """Compute a smooth daily target level for each date in the dataset.

    Method:
      1. Extract the midnight (00:00) reservoir level for each calendar day.
      2. Apply a 30-day centred rolling average to smooth out daily noise
         while preserving seasonal trends (spring filling, winter draw-down).
      3. Cap Haselholz targets at the physical upper bound (2800 mm).

    The resulting curve gives a fixed, physically meaningful reference for
    the end-of-day terminal constraint.  Using a fixed external target
    (rather than "return to today's start") prevents multi-week drift.

    Returns
    -------
    dict : date -> (target_lb_mm, target_lh_mm)
    """
    midnight = df[df['DateTime'].dt.time == pd.Timestamp('00:00').time()].copy()
    midnight = midnight.sort_values('DateTime').set_index('DateTime')

    lb_series = midnight['Bidmi_mm']
    lh_series = midnight['Haselholz_mm']

    target_lb = lb_series.rolling(window=ROLLING_WINDOW, center=True, min_periods=10).mean()
    target_lh = lh_series.rolling(window=ROLLING_WINDOW, center=True, min_periods=10).mean()

    # Fill the ~15-day edge windows where rolling average has fewer neighbours
    target_lb = target_lb.bfill().ffill()
    target_lh = target_lh.bfill().ffill()

    # Cap Haselholz: months where the smoothed average exceeds the physical
    # upper bound (May–August) cannot be reached — clip to the bound.
    target_lh = target_lh.clip(upper=HASELHOLZ_LEVEL_MAX)

    return {
        dt.date(): (float(lb), float(lh))
        for dt, lb, lh in zip(target_lb.index, target_lb.values, target_lh.values)
    }


# ---------------------------------------------------------------------------
# MODEL BUILDER
# ---------------------------------------------------------------------------

def build_model(N, demand, price, nat_inflow_b, nat_inflow_h,
                lb0, lh0, target_lb, target_lh):
    """Build and return a Pyomo ConcreteModel for one day.

    Parameters
    ----------
    N             : int          number of timesteps
    demand        : list[float]  Consumption_kW per step
    price         : list[float]  Spot_Price_CHF_MWh per step
    nat_inflow_b  : list[float]  natural inflow to Bidmi (mm/step)
    nat_inflow_h  : list[float]  natural inflow to Haselholz (mm/step)
    lb0, lh0      : float        starting levels — optimised end of previous day
                                 (actual observed level for the first day)
    target_lb     : float        seasonal target for Bidmi end-of-day level (mm)
    target_lh     : float        seasonal target for Haselholz end-of-day level (mm)
    """
    m = ConcreteModel()

    T     = list(range(N))
    T_ext = list(range(N + 1))
    m.T     = Set(initialize=T)
    m.T_ext = Set(initialize=T_ext)

    # --- Decision variable ---
    m.P = Var(m.T, domain=NonNegativeReals, bounds=(0.0, P_MAX_COMBINED))

    # --- Reservoir levels ---
    m.LB = Var(m.T_ext, bounds=(BIDMI_LEVEL_MIN, None))
    m.LH = Var(m.T_ext, bounds=(HASELHOLZ_LEVEL_MIN, None))
    m.LB[0].fix(lb0)
    m.LH[0].fix(lh0)

    # Hard upper bounds for t >= 1
    m.lb_hi = Constraint(
        [t for t in T_ext if t >= 1],
        rule=lambda m, t: m.LB[t] <= BIDMI_LEVEL_MAX
    )
    m.lh_hi = Constraint(
        [t for t in T_ext if t >= 1],
        rule=lambda m, t: m.LH[t] <= HASELHOLZ_LEVEL_MAX
    )

    # --- Spillage ---
    m.spill_b = Var(m.T, domain=NonNegativeReals)
    m.spill_h = Var(m.T, domain=NonNegativeReals)

    # --- Per-turbine rated maxima ---
    m.prod_max_bidmi = Constraint(m.T, rule=lambda m, t: m.P[t] * FRAC_B <= P_MAX_BIDMI)
    m.prod_max_hasel = Constraint(m.T, rule=lambda m, t: m.P[t] * FRAC_H <= P_MAX_HASELHOLZ)

    # --- Ramp constraints ---
    def ramp_up_rule(m, t):
        if t == 0: return Constraint.Skip
        return m.P[t] - m.P[t-1] <= RAMP_MAX

    def ramp_dn_rule(m, t):
        if t == 0: return Constraint.Skip
        return m.P[t-1] - m.P[t] <= RAMP_MAX

    m.ramp_up = Constraint(m.T, rule=ramp_up_rule)
    m.ramp_dn = Constraint(m.T, rule=ramp_dn_rule)

    # --- Water balance ---
    NET_COEFF_H = COEFF_B - COEFF_H

    m.wb_b = Constraint(m.T,
        rule=lambda m, t: m.LB[t+1] == m.LB[t] + nat_inflow_b[t]
                          - COEFF_B * m.P[t] - m.spill_b[t])
    m.wb_h = Constraint(m.T,
        rule=lambda m, t: m.LH[t+1] == m.LH[t] + nat_inflow_h[t]
                          + NET_COEFF_H * m.P[t] - m.spill_h[t])

    # --- Piecewise-linear terminal penalty ---
    # Bidmi: penalise |LB[N] - target_lb| beyond the free zone
    m.abs_dev_b = Var(domain=NonNegativeReals)
    m.abs_dev_b_pos = Constraint(expr=m.abs_dev_b >= m.LB[N] - target_lb)
    m.abs_dev_b_neg = Constraint(expr=m.abs_dev_b >= target_lb - m.LB[N])

    m.pw_b2 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE2_WIDTH))
    m.pw_b3 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE3_WIDTH))
    m.pw_b4 = Var(domain=NonNegativeReals)
    # Total paid zone >= deviation beyond free band
    m.pw_b_link = Constraint(
        expr=m.pw_b2 + m.pw_b3 + m.pw_b4 >= m.abs_dev_b - TERMINAL_FREE_ZONE
    )

    # Haselholz: same structure
    m.abs_dev_h = Var(domain=NonNegativeReals)
    m.abs_dev_h_pos = Constraint(expr=m.abs_dev_h >= m.LH[N] - target_lh)
    m.abs_dev_h_neg = Constraint(expr=m.abs_dev_h >= target_lh - m.LH[N])

    m.pw_h2 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE2_WIDTH))
    m.pw_h3 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE3_WIDTH))
    m.pw_h4 = Var(domain=NonNegativeReals)
    m.pw_h_link = Constraint(
        expr=m.pw_h2 + m.pw_h3 + m.pw_h4 >= m.abs_dev_h - TERMINAL_FREE_ZONE
    )

    # --- Objective ---
    m.obj = Objective(
        expr=(
            sum((demand[t] - m.P[t]) * TIMESTEP_HOURS * price[t] / 1000.0
                for t in m.T)
            + PW_SLOPE2 * (m.pw_b2 + m.pw_h2)
            + PW_SLOPE3 * (m.pw_b3 + m.pw_h3)
            + PW_SLOPE4 * (m.pw_b4 + m.pw_h4)
            + SPILL_PENALTY * (
                sum(m.spill_b[t] for t in m.T)
                + sum(m.spill_h[t] for t in m.T)
            )
        ),
        sense=minimize,
    )

    return m


# ---------------------------------------------------------------------------
# SOLVER
# ---------------------------------------------------------------------------

def get_solver():
    solver = SolverFactory('glpk')
    if not solver.available():
        print("ERROR: GLPK solver not found. Install it via: conda install glpk")
        sys.exit(1)
    return solver


# ---------------------------------------------------------------------------
# DAILY OPTIMISATION
# ---------------------------------------------------------------------------

def optimise_day(day_df, solver, lb0=None, lh0=None, target_lb=None, target_lh=None):
    """Solve the LP for one day. Returns day_df with Opt_* columns added.

    lb0 / lh0       : starting levels (chained from previous day; None = use actual)
    target_lb / lh  : seasonal target for end-of-day level (from compute_seasonal_targets)
    """
    day_df = day_df.reset_index(drop=True)
    N = len(day_df)

    demand       = day_df['Consumption_kW'].tolist()
    price        = day_df['Spot_Price_CHF_MWh'].tolist()
    nat_inflow_b = day_df['Natural_Inflow_B_mm'].tolist()
    nat_inflow_h = day_df['Natural_Inflow_H_mm'].tolist()

    if lb0 is None:
        lb0 = float(day_df.loc[0, 'Bidmi_mm'])
    if lh0 is None:
        lh0 = float(day_df.loc[0, 'Haselholz_mm'])
    if target_lb is None:
        target_lb = lb0
    if target_lh is None:
        target_lh = lh0

    model  = build_model(N, demand, price, nat_inflow_b, nat_inflow_h,
                         lb0, lh0, target_lb, target_lh)
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    if status not in ('optimal', 'feasible'):
        date_str = str(day_df.loc[0, 'DateTime'].date())
        print(f"  ERROR: {date_str} -- solver status '{status}'")
        for col in ['Opt_Production_kW', 'Opt_Bidmi_Production_kW',
                    'Opt_Haselholz_Production_kW', 'Opt_Bidmi_mm',
                    'Opt_Haselholz_mm', 'Opt_Network_Exchange_kW',
                    'Opt_Energy_Trading_CHF', 'Opt_Terminal_Dev_Bidmi_mm',
                    'Opt_Terminal_Dev_Haselholz_mm',
                    'Opt_Target_Bidmi_mm', 'Opt_Target_Haselholz_mm']:
            day_df[col] = float('nan')
        day_df._opt_lb_end = lb0
        day_df._opt_lh_end = lh0
        return day_df

    opt_p  = [value(model.P[t]) for t in range(N)]
    opt_lb = [value(model.LB[t]) for t in range(N)]
    opt_lh = [value(model.LH[t]) for t in range(N)]

    # Terminal deviation is now relative to the seasonal target
    dev_b = value(model.LB[N]) - target_lb
    dev_h = value(model.LH[N]) - target_lh

    day_df['Opt_Production_kW']             = opt_p
    day_df['Opt_Bidmi_Production_kW']       = [p * FRAC_B for p in opt_p]
    day_df['Opt_Haselholz_Production_kW']   = [p * FRAC_H for p in opt_p]
    day_df['Opt_Bidmi_mm']                  = opt_lb
    day_df['Opt_Haselholz_mm']              = opt_lh
    day_df['Opt_Network_Exchange_kW']       = [demand[t] - opt_p[t] for t in range(N)]
    day_df['Opt_Energy_Trading_CHF']        = [
        (demand[t] - opt_p[t]) * TIMESTEP_HOURS * price[t] / 1000.0
        for t in range(N)
    ]
    # Seasonal targets stored on every row (convenient for plotting)
    day_df['Opt_Target_Bidmi_mm']           = target_lb
    day_df['Opt_Target_Haselholz_mm']       = target_lh
    # Terminal deviation stored on the last row only
    day_df['Opt_Terminal_Dev_Bidmi_mm']     = 0.0
    day_df['Opt_Terminal_Dev_Haselholz_mm'] = 0.0
    day_df.loc[day_df.index[-1], 'Opt_Terminal_Dev_Bidmi_mm']     = dev_b
    day_df.loc[day_df.index[-1], 'Opt_Terminal_Dev_Haselholz_mm'] = dev_h

    day_df._opt_lb_end = value(model.LB[N])
    day_df._opt_lh_end = value(model.LH[N])
    day_df._dev_b      = dev_b
    day_df._dev_h      = dev_h
    return day_df


# ---------------------------------------------------------------------------
# FULL-YEAR LOOP
# ---------------------------------------------------------------------------

def optimise_all(df, target_date=None):
    solver = get_solver()

    print("Computing seasonal targets from midnight data...")
    season_targets = compute_seasonal_targets(df)

    days = sorted(df['DateTime'].dt.date.unique())

    if target_date is not None:
        days = [d for d in days if str(d) == target_date]
        if not days:
            print(f"ERROR: date {target_date} not found in results.csv")
            sys.exit(1)

    results    = []
    prev_lb    = None   # chained starting level (None = use actual on first day)
    prev_lh    = None

    for i, day in enumerate(days, 1):
        day_df = df[df['DateTime'].dt.date == day].copy()

        t_lb, t_lh = season_targets.get(day, (None, None))

        day_df = optimise_day(day_df, solver,
                              lb0=prev_lb, lh0=prev_lh,
                              target_lb=t_lb, target_lh=t_lh)

        prev_lb = getattr(day_df, '_opt_lb_end', None)
        prev_lh = getattr(day_df, '_opt_lh_end', None)

        cost        = day_df['Opt_Energy_Trading_CHF'].sum()
        actual_cost = day_df['Energy_Trading_CHF'].sum()
        dev_b       = getattr(day_df, '_dev_b', float('nan'))
        dev_h       = getattr(day_df, '_dev_h', float('nan'))
        gain        = actual_cost - cost   # positive = opt better, negative = actual better

        if gain >= 0:
            gain_str = f"opt saves {gain:+.2f}"
        else:
            gain_str = f"actual better by {-gain:.2f}"

        print(f"[{i}/{len(days)}] {day}  "
              f"tgt B={t_lb:.0f} H={t_lh:.0f}  "
              f"actual {actual_cost:+.2f} CHF  opt {cost:+.2f} CHF  "
              f"[{gain_str}]  "
              f"dLB={dev_b:+.1f}mm dLH={dev_h:+.1f}mm")
        results.append(day_df)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    root = os.path.dirname(__file__)
    results_path = os.path.join(root, RESULTS_FILENAME)

    if not os.path.exists(results_path):
        print(f"ERROR: '{RESULTS_FILENAME}' not found.")
        print("  Run  python run_model.py  first.")
        sys.exit(1)

    target_date = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Loading: {results_path}")
    df = pd.read_csv(results_path, parse_dates=['DateTime'])
    print(f"  {len(df)} rows, {df['DateTime'].dt.date.nunique()} days")
    if target_date:
        print(f"  Optimising single day: {target_date}")
    print()

    opt_df = optimise_all(df, target_date)

    opt_df['Opt_Total_Energy_Cost_CHF'] = opt_df['Opt_Energy_Trading_CHF'].cumsum()

    output_path = os.path.join(root, OUTPUT_FILENAME)
    opt_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    total_actual = opt_df['Energy_Trading_CHF'].sum()
    total_opt    = opt_df['Opt_Energy_Trading_CHF'].sum()
    total_gain   = total_actual - total_opt   # positive = opt better overall

    # Count days where optimizer was better vs worse
    daily_actual = opt_df.groupby(opt_df['DateTime'].dt.date)['Energy_Trading_CHF'].sum()
    daily_opt    = opt_df.groupby(opt_df['DateTime'].dt.date)['Opt_Energy_Trading_CHF'].sum()
    days_opt_better    = (daily_actual > daily_opt).sum()
    days_actual_better = (daily_actual < daily_opt).sum()

    print(f"\nFull-period summary:")
    print(f"  Actual cost   : {total_actual:+.2f} CHF  (negative = net seller)")
    print(f"  Optimised cost: {total_opt:+.2f} CHF  (negative = net seller)")
    print(f"  Optimizer gain: {total_gain:+.2f} CHF  (positive = opt saves money vs actual)")
    print(f"  Days opt better   : {days_opt_better}")
    print(f"  Days actual better: {days_actual_better}  (opt constrained by terminal target)")


if __name__ == '__main__':
    main()
