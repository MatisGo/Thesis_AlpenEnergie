"""
Hydro scheduling LP optimiser
==============================

Reads matis_2025_.xlsx directly.

For each calendar day, solves an LP to find the optimal per-turbine
production schedule that minimises total energy cost (net grid purchases
at spot price).

Turbines
--------
  M2 — downstream of Bidmi reservoir       (up to 1700 kW)
  M1 — downstream of Haselholz reservoir   (up to 1156 kW)

M2 discharge cascades 100 % into the Haselholz reservoir.
M1 and M2 are fully independent (no fixed ratio).

Water balance (per 5-min timestep)
-----------------------------------
  ΔLB[t] = (Inflow_B[t] - Q_M2[t]) / BIDMI_LS_PER_MM
  ΔLH[t] = (Inflow_H[t] + Q_M2[t] - Q_M1[t]) / HASELHOLZ_LS_PER_MM

  Q_M2[t] [l/s] = P_M2[t] [kW] / M2_KW_PER_LS
  Q_M1[t] [l/s] = P_M1[t] [kW] / M1_KW_PER_LS

  Inflow_B = Excel column Q  [l/s]
  Inflow_H = Excel column P  [l/s]

Day chaining
------------
The optimised end-of-day level becomes the starting level for the next day.

Seasonal targets
----------------
Two complementary constraints:

1. Midnight terminal penalty (strong):
   30-day centred rolling average of actual midnight levels.
   Piecewise-linear one-sided penalty on end-of-day deficit.

2. Intra-day soft floor (very gentle, 0.05 CHF/mm):
   For every 5-min slot, 30-day centred rolling average of the actual
   level at that exact time-of-day.  Penalises level below the floor —
   caps at physical reservoir maxima so the target is always reachable.

Usage
-----
  python optimise.py               # all days in Excel
  python optimise.py 2025-06-15   # single day
"""

import os
import sys

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals,
    Constraint, Objective, minimize, maximize, value, SolverFactory,
)

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------

# Turbine conversion: kW per l/s
M2_KW_PER_LS = 4.537   # M2 = Bidmi turbine
M1_KW_PER_LS = 1.634   # M1 = Haselholz turbine

# Reservoir level sensitivity: l/s that causes 1 mm level change per timestep
BIDMI_LS_PER_MM     = 8.33
HASELHOLZ_LS_PER_MM = 4.39

# Effective LP coefficients  (mm of level change per kW of production)
# Bidmi drops by this much per kW of M2
COEFF_M2_BIDMI     = 1.0 / (M2_KW_PER_LS * BIDMI_LS_PER_MM)
# Haselholz rises by this much per kW of M2 (cascade)
COEFF_M2_CASCADE   = 1.0 / (M2_KW_PER_LS * HASELHOLZ_LS_PER_MM)
# Haselholz drops by this much per kW of M1
COEFF_M1_HASELHOLZ = 1.0 / (M1_KW_PER_LS * HASELHOLZ_LS_PER_MM)

TIMESTEP_HOURS = 5 / 60   # 5-minute resolution

P_MAX_M2 = 1920.0   # kW
P_MAX_M1 = 1156.0   # kW

RAMP_MAX    = 250.0   # kW  — max combined (M1+M2) change over RAMP_WINDOW
RAMP_WINDOW =   3     # timesteps = 15 minutes

BIDMI_LEVEL_MIN     = 1000.0   # mm
BIDMI_LEVEL_MAX     = 2200.0   # mm
HASELHOLZ_LEVEL_MIN =  600.0   # mm
HASELHOLZ_LEVEL_MAX = 2800.0   # mm

# ---------------------------------------------------------------------------
# SEASONAL TARGET PARAMETERS
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 30   # days for smoothing midnight levels

# ---------------------------------------------------------------------------
# PIECEWISE-LINEAR TERMINAL PENALTY
# ---------------------------------------------------------------------------
#
#  |deviation from seasonal target|   penalty rate
#  0   – 100 mm                       0 CHF/mm   (free zone)
#  100 – 200 mm                       PW_SLOPE2
#  200 – 300 mm                       PW_SLOPE3
#  > 300 mm                           PW_SLOPE4
#
TERMINAL_FREE_ZONE = 100.0
PW_ZONE2_WIDTH     = 100.0
PW_ZONE3_WIDTH     = 100.0
PW_SLOPE2          =   2.0   # CHF/mm
PW_SLOPE3          =   8.0   # CHF/mm
PW_SLOPE4          =  25.0   # CHF/mm

SPILL_PENALTY = 1.0   # CHF/mm

# Intra-day soft floor penalty — very gentle, only guides the shape of the
# level curve during the day.  Much softer than the terminal penalty (2 CHF/mm).
INTRADAY_FLOOR_PENALTY = 0.05   # CHF/mm per timestep

# Ratio used to estimate cascade energy lost when Bidmi spills:
# spilled water that passed through M2 would also have passed through M1
TURBINE_CASCADE_RATIO = 1.560   # historical M1/M2 production ratio

DATA_FILENAME   = 'matis_2025_.xlsx'
OUTPUT_FILENAME = 'optimised_results.csv'


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data(path):
    """Read matis_2025_.xlsx and return a clean DataFrame.

    Excel column layout (0-indexed after skipping 3 header rows):
      A=0   DateTime
      F=5   Consumption_kW
      I=8   Haselholz_m   (reservoir level, metres)
      J=9   Bidmi_m       (reservoir level, metres)
      K=10  Spot_Price_CHF_MWh
      L=11  Ref_M1_kW     (reference Haselholz production)
      M=12  Ref_M2_kW     (reference Bidmi production)
      P=15  Haselholz_Inflow_ls   (natural inflow, l/s)
      Q=16  Bidmi_Inflow_ls       (natural inflow, l/s)
    """
    raw = pd.read_excel(path, header=None, skiprows=3)

    df = raw.iloc[:, [0, 5, 8, 9, 10, 11, 12, 15, 16]].copy()
    df.columns = [
        'DateTime',
        'Consumption_kW',
        'Haselholz_m',
        'Bidmi_m',
        'Spot_Price_CHF_MWh',
        'Ref_M1_kW',
        'Ref_M2_kW',
        'Haselholz_Inflow_ls',
        'Bidmi_Inflow_ls',
    ]

    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    numeric_cols = [
        'Consumption_kW', 'Haselholz_m', 'Bidmi_m',
        'Spot_Price_CHF_MWh', 'Ref_M2_kW', 'Ref_M1_kW',
        'Haselholz_Inflow_ls', 'Bidmi_Inflow_ls',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # Convert reservoir levels m -> mm
    df['Bidmi_mm']     = df['Bidmi_m']     * 1000
    df['Haselholz_mm'] = df['Haselholz_m'] * 1000

    # Reference energy trading profit (positive = selling surplus, negative = buying deficit)
    ref_prod = df['Ref_M2_kW'] + df['Ref_M1_kW']
    df['Ref_Network_Exchange_kW'] = df['Consumption_kW'] - ref_prod
    df['Ref_Energy_Trading_CHF']  = (
        (ref_prod - df['Consumption_kW']) * TIMESTEP_HOURS * df['Spot_Price_CHF_MWh'] / 1000
    )

    return df


# ---------------------------------------------------------------------------
# SEASONAL TARGET COMPUTATION
# ---------------------------------------------------------------------------

def compute_seasonal_targets(df):
    """30-day centred rolling average of actual midnight reservoir levels.

    Returns dict: date -> (target_bidmi_mm, target_haselholz_mm)
    """
    midnight = df[df['DateTime'].dt.time == pd.Timestamp('00:00').time()].copy()
    midnight = midnight.sort_values('DateTime').set_index('DateTime')

    target_lb = midnight['Bidmi_mm'].rolling(
        window=ROLLING_WINDOW, center=True, min_periods=10).mean()
    target_lh = midnight['Haselholz_mm'].rolling(
        window=ROLLING_WINDOW, center=True, min_periods=10).mean()

    target_lb = target_lb.bfill().ffill()
    target_lh = target_lh.bfill().ffill()
    target_lh = target_lh.clip(upper=HASELHOLZ_LEVEL_MAX)

    return {
        dt.date(): (float(lb), float(lh))
        for dt, lb, lh in zip(target_lb.index, target_lb.values, target_lh.values)
    }


# ---------------------------------------------------------------------------
# INTRA-DAY SEASONAL FLOOR COMPUTATION
# ---------------------------------------------------------------------------

def compute_intraday_targets(df):
    """30-day centred rolling average of actual level at each 5-min slot.

    For every (date, time-of-day) pair, the target is the average actual
    reservoir level across the ±15 days surrounding that date at the same
    clock time.  Targets are capped at the physical reservoir maxima so
    the optimizer can always reach them.

    Returns dict: date -> (floor_b_array, floor_h_array)
                  each array has length = number of timesteps that day.
    """
    df_copy = df.copy()
    df_copy['date']     = df_copy['DateTime'].dt.date
    df_copy['time_idx'] = df_copy.groupby('date').cumcount()

    # Pivot: rows = date, columns = time-of-day index
    pivot_b = df_copy.pivot(index='date', columns='time_idx', values='Bidmi_mm')
    pivot_h = df_copy.pivot(index='date', columns='time_idx', values='Haselholz_mm')

    # 30-day centred rolling mean along the date axis for every time slot
    floor_b = pivot_b.rolling(window=ROLLING_WINDOW, center=True, min_periods=10).mean()
    floor_h = pivot_h.rolling(window=ROLLING_WINDOW, center=True, min_periods=10).mean()

    floor_b = floor_b.bfill().ffill()
    floor_h = floor_h.bfill().ffill()

    # Fill any remaining NaN (e.g. DST day extra slots, edge of dataset)
    # with the physical minimum — effectively no constraint on those steps
    floor_b = floor_b.fillna(BIDMI_LEVEL_MIN)
    floor_h = floor_h.fillna(HASELHOLZ_LEVEL_MIN)

    # Cap at physical maxima — target must always be reachable
    floor_b = floor_b.clip(upper=BIDMI_LEVEL_MAX)
    floor_h = floor_h.clip(upper=HASELHOLZ_LEVEL_MAX)

    return {
        date: (floor_b.loc[date].values.tolist(),
               floor_h.loc[date].values.tolist())
        for date in floor_b.index
    }


# ---------------------------------------------------------------------------
# MODEL BUILDER
# ---------------------------------------------------------------------------

def build_model(N, demand, price, inflow_b, inflow_h,
                lb0, lh0, target_lb, target_lh,
                floor_lb, floor_lh):
    """Build and return a Pyomo LP for one day.

    Parameters
    ----------
    N          : int          number of timesteps (288 @ 5-min)
    demand     : list[float]  Consumption_kW per step
    price      : list[float]  Spot_Price_CHF_MWh per step
    inflow_b   : list[float]  Bidmi natural inflow [l/s] per step
    inflow_h   : list[float]  Haselholz natural inflow [l/s] per step
                              (does NOT include M2 discharge — added in water balance)
    lb0, lh0   : float        starting reservoir levels [mm]
    target_lb  : float        seasonal target for Bidmi end-of-day level [mm]
    target_lh  : float        seasonal target for Haselholz end-of-day level [mm]
    floor_lb   : list[float]  intra-day soft floor for Bidmi [mm], length N
    floor_lh   : list[float]  intra-day soft floor for Haselholz [mm], length N
    """
    m = ConcreteModel()

    T     = list(range(N))
    T_ext = list(range(N + 1))
    m.T     = Set(initialize=T)
    m.T_ext = Set(initialize=T_ext)

    # --- Decision variables: independent production per turbine ---
    m.P_M2 = Var(m.T, domain=NonNegativeReals, bounds=(0.0, P_MAX_M2))   # Bidmi
    m.P_M1 = Var(m.T, domain=NonNegativeReals, bounds=(0.0, P_MAX_M1))   # Haselholz

    # --- Reservoir levels ---
    m.LB = Var(m.T_ext, bounds=(BIDMI_LEVEL_MIN, None))
    m.LH = Var(m.T_ext, bounds=(HASELHOLZ_LEVEL_MIN, None))
    m.LB[0].fix(lb0)
    m.LH[0].fix(lh0)

    # Hard upper bounds for t >= 1
    m.lb_hi = Constraint(
        [t for t in T_ext if t >= 1],
        rule=lambda m, t: m.LB[t] <= BIDMI_LEVEL_MAX)
    m.lh_hi = Constraint(
        [t for t in T_ext if t >= 1],
        rule=lambda m, t: m.LH[t] <= HASELHOLZ_LEVEL_MAX)

    # --- Spillage (overflow relief) ---
    m.spill_b = Var(m.T, domain=NonNegativeReals)
    m.spill_h = Var(m.T, domain=NonNegativeReals)

    # --- Ramp constraints on combined production ---
    def ramp_up_rule(m, t):
        if t < RAMP_WINDOW: return Constraint.Skip
        return (m.P_M2[t] + m.P_M1[t]) - (m.P_M2[t - RAMP_WINDOW] + m.P_M1[t - RAMP_WINDOW]) <= RAMP_MAX

    def ramp_dn_rule(m, t):
        if t < RAMP_WINDOW: return Constraint.Skip
        return (m.P_M2[t - RAMP_WINDOW] + m.P_M1[t - RAMP_WINDOW]) - (m.P_M2[t] + m.P_M1[t]) <= RAMP_MAX

    m.ramp_up = Constraint(m.T, rule=ramp_up_rule)
    m.ramp_dn = Constraint(m.T, rule=ramp_dn_rule)

    # --- Water balance ---
    #
    # Bidmi:
    #   LB[t+1] = LB[t] + (Inflow_B[t] - Q_M2[t]) / BIDMI_LS_PER_MM - spill_b[t]
    #   Q_M2[t] = P_M2[t] / M2_KW_PER_LS
    #   => LB[t+1] = LB[t] + Inflow_B[t]/BIDMI_LS_PER_MM
    #                       - COEFF_M2_BIDMI * P_M2[t]
    #                       - spill_b[t]
    #
    m.wb_b = Constraint(m.T, rule=lambda m, t:
        m.LB[t+1] == m.LB[t]
                     + inflow_b[t] / BIDMI_LS_PER_MM
                     - COEFF_M2_BIDMI * m.P_M2[t]
                     - m.spill_b[t])

    # Haselholz (receives M2 discharge as cascade input):
    #   LH[t+1] = LH[t] + (Inflow_H[t] + Q_M2[t] - Q_M1[t]) / HASELHOLZ_LS_PER_MM - spill_h[t]
    #   => LH[t+1] = LH[t] + Inflow_H[t]/HASELHOLZ_LS_PER_MM
    #                       + COEFF_M2_CASCADE   * P_M2[t]
    #                       - COEFF_M1_HASELHOLZ * P_M1[t]
    #                       - spill_h[t]
    #
    m.wb_h = Constraint(m.T, rule=lambda m, t:
        m.LH[t+1] == m.LH[t]
                     + inflow_h[t] / HASELHOLZ_LS_PER_MM
                     + COEFF_M2_CASCADE   * m.P_M2[t]
                     - COEFF_M1_HASELHOLZ * m.P_M1[t]
                     - m.spill_h[t])

    # --- Intra-day soft floor (one-sided, very gentle) ---
    # For each timestep t, penalise max(0, floor[t] - L[t]).
    # floor_slack[t] >= 0  and  floor_slack[t] >= floor[t] - L[t]
    # => optimizer sets floor_slack[t] = max(0, floor[t] - L[t])
    m.floor_slack_b = Var(m.T, domain=NonNegativeReals)
    m.floor_slack_h = Var(m.T, domain=NonNegativeReals)

    m.floor_con_b = Constraint(m.T, rule=lambda m, t:
        m.floor_slack_b[t] >= floor_lb[t] - m.LB[t])
    m.floor_con_h = Constraint(m.T, rule=lambda m, t:
        m.floor_slack_h[t] >= floor_lh[t] - m.LH[t])

    # --- Piecewise-linear terminal penalty (one-sided: below target only) ---
    # Being above the seasonal target is free — water stored is available tomorrow.
    # Being below is penalised in escalating zones.
    #
    # Bidmi: penalise max(0, target_lb - LB[N]) beyond the free zone
    m.dev_below_b = Var(domain=NonNegativeReals)
    m.dev_below_b_con = Constraint(expr=m.dev_below_b >= target_lb - m.LB[N])
    m.pw_b2 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE2_WIDTH))
    m.pw_b3 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE3_WIDTH))
    m.pw_b4 = Var(domain=NonNegativeReals)
    m.pw_b_link = Constraint(
        expr=m.pw_b2 + m.pw_b3 + m.pw_b4 >= m.dev_below_b - TERMINAL_FREE_ZONE)

    # Haselholz: same structure
    m.dev_below_h = Var(domain=NonNegativeReals)
    m.dev_below_h_con = Constraint(expr=m.dev_below_h >= target_lh - m.LH[N])
    m.pw_h2 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE2_WIDTH))
    m.pw_h3 = Var(domain=NonNegativeReals, bounds=(0.0, PW_ZONE3_WIDTH))
    m.pw_h4 = Var(domain=NonNegativeReals)
    m.pw_h_link = Constraint(
        expr=m.pw_h2 + m.pw_h3 + m.pw_h4 >= m.dev_below_h - TERMINAL_FREE_ZONE)

    # --- Objective: maximise profit ---
    # Profit  = revenue from selling surplus − cost of buying deficit
    #         = (production − demand) × price × Δt / 1000
    # Penalties reduce profit: terminal deficit penalty + spill penalty.
    # Note: being above the seasonal target at end of day carries no penalty.
    m.obj = Objective(
        expr=(
            sum((m.P_M2[t] + m.P_M1[t] - demand[t]) * TIMESTEP_HOURS * price[t] / 1000.0
                for t in m.T)
            - PW_SLOPE2 * (m.pw_b2 + m.pw_h2)
            - PW_SLOPE3 * (m.pw_b3 + m.pw_h3)
            - PW_SLOPE4 * (m.pw_b4 + m.pw_h4)
            - SPILL_PENALTY * (
                sum(m.spill_b[t] for t in m.T)
                + sum(m.spill_h[t] for t in m.T)
            )
            - INTRADAY_FLOOR_PENALTY * (
                sum(m.floor_slack_b[t] for t in m.T)
                + sum(m.floor_slack_h[t] for t in m.T)
            )
        ),
        sense=maximize,
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

def optimise_day(day_df, solver, lb0=None, lh0=None,
                 target_lb=None, target_lh=None,
                 floor_lb=None, floor_lh=None):
    """Solve the LP for one day. Returns day_df with Opt_* columns added.

    lb0 / lh0        : starting levels (chained from previous day; None = actual)
    target_lb / lh   : seasonal end-of-day target [mm]
    floor_lb / lh    : intra-day soft floor per timestep [mm], length N
    """
    day_df = day_df.reset_index(drop=True)
    N = len(day_df)

    demand   = day_df['Consumption_kW'].tolist()
    price    = day_df['Spot_Price_CHF_MWh'].tolist()
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    if lb0 is None:
        lb0 = float(day_df.loc[0, 'Bidmi_mm'])
    if lh0 is None:
        lh0 = float(day_df.loc[0, 'Haselholz_mm'])
    if target_lb is None:
        target_lb = lb0
    if target_lh is None:
        target_lh = lh0
    # Ensure floor arrays are exactly length N (handles None, DST days, edge cases)
    def _fit(arr, n, fill):
        if arr is None:
            return [fill] * n
        arr = list(arr)[:n]            # truncate if too long
        arr += [fill] * (n - len(arr)) # pad if too short
        return arr

    floor_lb = _fit(floor_lb, N, BIDMI_LEVEL_MIN)
    floor_lh = _fit(floor_lh, N, HASELHOLZ_LEVEL_MIN)

    model  = build_model(N, demand, price, inflow_b, inflow_h,
                         lb0, lh0, target_lb, target_lh,
                         floor_lb, floor_lh)
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    opt_cols = [
        'Opt_M2_kW', 'Opt_M1_kW', 'Opt_Production_kW',
        'Opt_Bidmi_mm', 'Opt_Haselholz_mm',
        'Opt_Energy_Trading_CHF',
        'Opt_Target_Bidmi_mm', 'Opt_Target_Haselholz_mm',
        'Opt_Terminal_Dev_Bidmi_mm', 'Opt_Terminal_Dev_Haselholz_mm',
        'Opt_Spill_Bidmi_mm', 'Opt_Spill_Haselholz_mm',
        'Opt_Spill_Bidmi_kWh', 'Opt_Spill_Haselholz_kWh',
    ]

    if status not in ('optimal', 'feasible'):
        date_str = str(day_df.loc[0, 'DateTime'].date())
        print(f"  ERROR: {date_str} -- solver status '{status}'")
        for col in opt_cols:
            day_df[col] = float('nan')
        day_df._opt_lb_end = lb0
        day_df._opt_lh_end = lh0
        return day_df

    opt_m2 = [value(model.P_M2[t]) for t in range(N)]
    opt_m1 = [value(model.P_M1[t]) for t in range(N)]
    opt_p  = [opt_m2[t] + opt_m1[t] for t in range(N)]
    opt_lb = [value(model.LB[t]) for t in range(N)]
    opt_lh = [value(model.LH[t]) for t in range(N)]

    dev_b = value(model.LB[N]) - target_lb
    dev_h = value(model.LH[N]) - target_lh

    # --- Spillage: extract LP variables and convert to energy equivalent ---
    opt_spill_b = [value(model.spill_b[t]) for t in range(N)]
    opt_spill_h = [value(model.spill_h[t]) for t in range(N)]

    # Bidmi spill energy loss:
    #   spill_b [mm] -> l/s via BIDMI_LS_PER_MM -> kW at M2 via M2_KW_PER_LS
    #   That same water would also have produced energy at M1 (cascade ratio 1.560)
    #   Total loss = M2 energy * (1 + TURBINE_CASCADE_RATIO)
    spill_b_kwh = [
        s * BIDMI_LS_PER_MM * M2_KW_PER_LS * TIMESTEP_HOURS * (1 + TURBINE_CASCADE_RATIO)
        for s in opt_spill_b
    ]

    # Haselholz spill energy loss:
    #   spill_h [mm] -> l/s via HASELHOLZ_LS_PER_MM -> kW at M1 via M1_KW_PER_LS
    spill_h_kwh = [
        s * HASELHOLZ_LS_PER_MM * M1_KW_PER_LS * TIMESTEP_HOURS
        for s in opt_spill_h
    ]

    day_df['Opt_M2_kW']         = opt_m2
    day_df['Opt_M1_kW']         = opt_m1
    day_df['Opt_Production_kW'] = opt_p
    day_df['Opt_Bidmi_mm']      = opt_lb
    day_df['Opt_Haselholz_mm']  = opt_lh
    # Positive = profit (selling surplus), negative = cost (buying deficit)
    day_df['Opt_Energy_Trading_CHF'] = [
        (opt_p[t] - demand[t]) * TIMESTEP_HOURS * price[t] / 1000.0
        for t in range(N)
    ]
    day_df['Opt_Target_Bidmi_mm']           = target_lb
    day_df['Opt_Target_Haselholz_mm']       = target_lh
    day_df['Opt_Terminal_Dev_Bidmi_mm']     = 0.0
    day_df['Opt_Terminal_Dev_Haselholz_mm'] = 0.0
    day_df.loc[day_df.index[-1], 'Opt_Terminal_Dev_Bidmi_mm']     = dev_b
    day_df.loc[day_df.index[-1], 'Opt_Terminal_Dev_Haselholz_mm'] = dev_h
    day_df['Opt_Spill_Bidmi_mm']      = opt_spill_b
    day_df['Opt_Spill_Haselholz_mm']  = opt_spill_h
    day_df['Opt_Spill_Bidmi_kWh']     = spill_b_kwh
    day_df['Opt_Spill_Haselholz_kWh'] = spill_h_kwh

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

    print("Computing intra-day seasonal floors (30-day rolling per time slot)...")
    intraday_targets = compute_intraday_targets(df)

    days = sorted(df['DateTime'].dt.date.unique())

    if target_date is not None:
        days = [d for d in days if str(d) == target_date]
        if not days:
            print(f"ERROR: date {target_date} not found in data")
            sys.exit(1)

    results = []
    prev_lb = None
    prev_lh = None

    for i, day in enumerate(days, 1):
        day_df = df[df['DateTime'].dt.date == day].copy()

        t_lb, t_lh           = season_targets.get(day, (None, None))
        f_lb, f_lh           = intraday_targets.get(day, (None, None))

        day_df = optimise_day(day_df, solver,
                              lb0=prev_lb, lh0=prev_lh,
                              target_lb=t_lb, target_lh=t_lh,
                              floor_lb=f_lb, floor_lh=f_lh)

        prev_lb = getattr(day_df, '_opt_lb_end', None)
        prev_lh = getattr(day_df, '_opt_lh_end', None)

        opt_profit = day_df['Opt_Energy_Trading_CHF'].sum()
        ref_profit = day_df['Ref_Energy_Trading_CHF'].sum()
        dev_b      = getattr(day_df, '_dev_b', float('nan'))
        dev_h      = getattr(day_df, '_dev_h', float('nan'))
        gain       = opt_profit - ref_profit
        spill_b_e  = day_df['Opt_Spill_Bidmi_kWh'].sum()
        spill_h_e  = day_df['Opt_Spill_Haselholz_kWh'].sum()

        gain_str  = f"opt +{gain:.2f}" if gain >= 0 else f"ref better by {-gain:.2f}"
        spill_str = (f"  SPILL B={spill_b_e:.0f} H={spill_h_e:.0f} kWh"
                     if spill_b_e + spill_h_e > 0.1 else "")

        print(f"[{i}/{len(days)}] {day}  "
              f"tgt B={t_lb:.0f} H={t_lh:.0f}  "
              f"ref {ref_profit:+.2f} CHF  opt {opt_profit:+.2f} CHF  "
              f"[{gain_str}]  "
              f"dLB={dev_b:+.1f}mm dLH={dev_h:+.1f}mm"
              f"{spill_str}")

        results.append(day_df)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    root = os.path.dirname(__file__)
    data_path = os.path.join(root, DATA_FILENAME)

    if not os.path.exists(data_path):
        print(f"ERROR: '{DATA_FILENAME}' not found.")
        sys.exit(1)

    target_date = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Loading: {data_path}")
    df = load_data(data_path)
    print(f"  {len(df)} rows, {df['DateTime'].dt.date.nunique()} days")
    if target_date:
        print(f"  Optimising single day: {target_date}")
    print()

    opt_df = optimise_all(df, target_date)

    opt_df['Opt_Total_Energy_Cost_CHF'] = opt_df['Opt_Energy_Trading_CHF'].cumsum()

    output_path = os.path.join(root, OUTPUT_FILENAME)
    opt_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    total_ref  = opt_df['Ref_Energy_Trading_CHF'].sum()
    total_opt  = opt_df['Opt_Energy_Trading_CHF'].sum()
    total_gain = total_opt - total_ref   # positive = opt earns more

    daily_ref = opt_df.groupby(opt_df['DateTime'].dt.date)['Ref_Energy_Trading_CHF'].sum()
    daily_opt = opt_df.groupby(opt_df['DateTime'].dt.date)['Opt_Energy_Trading_CHF'].sum()
    days_opt_better = (daily_opt > daily_ref).sum()
    days_ref_better = (daily_opt < daily_ref).sum()

    # Cumulative energy produced (kWh)
    ref_energy_kwh = ((opt_df['Ref_M2_kW'] + opt_df['Ref_M1_kW']) * TIMESTEP_HOURS).sum()
    opt_energy_kwh = opt_df['Opt_Production_kW'].sum() * TIMESTEP_HOURS
    energy_diff    = opt_energy_kwh - ref_energy_kwh

    # Spillage energy losses
    total_spill_b_kwh = opt_df['Opt_Spill_Bidmi_kWh'].sum()
    total_spill_h_kwh = opt_df['Opt_Spill_Haselholz_kWh'].sum()
    total_spill_kwh   = total_spill_b_kwh + total_spill_h_kwh

    print(f"\nFull-period summary:")
    print(f"  Reference profit: {total_ref:+.2f} CHF  (positive = net seller)")
    print(f"  Optimised profit: {total_opt:+.2f} CHF  (positive = net seller)")
    print(f"  Optimizer gain  : {total_gain:+.2f} CHF  (positive = opt earns more than reference)")
    print(f"  Days opt better : {days_opt_better}")
    print(f"  Days ref better : {days_ref_better}")
    print()
    print(f"  Reference energy produced: {ref_energy_kwh:,.0f} kWh  ({ref_energy_kwh/1000:,.1f} MWh)")
    print(f"  Optimised energy produced: {opt_energy_kwh:,.0f} kWh  ({opt_energy_kwh/1000:,.1f} MWh)")
    print(f"  Difference               : {energy_diff:+,.0f} kWh  ({energy_diff/1000:+,.1f} MWh)"
          f"  ({'opt produces more' if energy_diff >= 0 else 'ref produces more'})")
    print()
    print(f"  Spillage energy losses (optimised schedule):")
    print(f"    Bidmi     : {total_spill_b_kwh:,.0f} kWh  (M2 loss + M1 cascade @ ratio 1.560)")
    print(f"    Haselholz : {total_spill_h_kwh:,.0f} kWh  (M1 loss)")
    print(f"    Total     : {total_spill_kwh:,.0f} kWh  ({total_spill_kwh/1000:,.1f} MWh)")


if __name__ == '__main__':
    main()
