"""
hydro_constants.py
==================
Physical constants, data loading, and shared utilities for all hydro
scheduling modes.  Import everything from here — never copy constants
across mode files.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Turbine efficiencies
# ---------------------------------------------------------------------------
M2_KW_PER_LS = 4.537        # kW per l/s  (M2 = Bidmi turbine)
M1_KW_PER_LS = 1.634        # kW per l/s  (M1 = Haselholz turbine)

# ---------------------------------------------------------------------------
# Reservoir hydraulics
# ---------------------------------------------------------------------------
BIDMI_LS_PER_MM     = 8.33  # l/s that changes Bidmi level by 1 mm per timestep
HASELHOLZ_LS_PER_MM = 4.39  # l/s that changes Haselholz level by 1 mm per timestep

# Level change per kW of production (mm/kW per timestep)
COEFF_M2_BIDMI     = 1.0 / (M2_KW_PER_LS * BIDMI_LS_PER_MM)      # Bidmi drops
COEFF_M2_CASCADE   = 1.0 / (M2_KW_PER_LS * HASELHOLZ_LS_PER_MM)  # Haselholz rises (cascade)
COEFF_M1_HASELHOLZ = 1.0 / (M1_KW_PER_LS * HASELHOLZ_LS_PER_MM)  # Haselholz drops

# ---------------------------------------------------------------------------
# Physical reservoir bounds  [mm]
# ---------------------------------------------------------------------------
BIDMI_LEVEL_MIN     = 1000.0
BIDMI_LEVEL_MAX     = 2200.0
HASELHOLZ_LEVEL_MIN =  600.0
HASELHOLZ_LEVEL_MAX = 2800.0

BIDMI_RANGE     = BIDMI_LEVEL_MAX     - BIDMI_LEVEL_MIN    # 1200 mm
HASELHOLZ_RANGE = HASELHOLZ_LEVEL_MAX - HASELHOLZ_LEVEL_MIN  # 2200 mm

# ---------------------------------------------------------------------------
# Turbine limits and ramp rates
# ---------------------------------------------------------------------------
P_MAX_M2    = 1920.0   # kW
P_MAX_M1    = 1156.0   # kW
RAMP_MAX    =  200.0   # kW per RAMP_WINDOW timesteps
RAMP_WINDOW =    3     # 3 × 5 min = 15 min

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
TIMESTEP_HOURS = 5 / 60   # 5-minute resolution

# ---------------------------------------------------------------------------
# Grid exchange peak tariff  (Leistungsentgelt)
# ---------------------------------------------------------------------------
PEAK_TARIFF_EUR_KW = 10.0   # EUR per kW of monthly peak import at grid connection

# ---------------------------------------------------------------------------
# Spill energy loss factor
# ---------------------------------------------------------------------------
# Bidmi spill loses M2 energy AND the subsequent M1 cascade energy
TURBINE_CASCADE_RATIO = 1.560   # historical M1/M2 production ratio

# ---------------------------------------------------------------------------
# Seasonal target rolling window
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 30   # days

# ---------------------------------------------------------------------------
# Data file
# ---------------------------------------------------------------------------
DATA_FILENAME = 'matis_2025_.xlsx'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Read matis_2025_.xlsx and return a clean DataFrame."""
    raw = pd.read_excel(path, header=None, skiprows=3, engine='calamine')

    # Column layout (0-indexed, after skiprows=3):
    #   A=0 DateTime  E=4 Forecast_kW  G=6 Consumption_kW
    #   J=9 Haselholz_m  K=10 Bidmi_m
    #   L=11 Day_Ahead_Price_EUR_MWh  M=12 Intra_Day_Price_EUR_MWh
    #   N=13 Ref_M1_kW  O=14 Ref_M2_kW
    #   R=17 Haselholz_Inflow_ls  S=18 Bidmi_Inflow_ls
    df = raw.iloc[:, [0, 4, 6, 9, 10, 11, 12, 13, 14, 17, 18]].copy()
    df.columns = [
        'DateTime', 'Forecast_kW', 'Consumption_kW',
        'Haselholz_m', 'Bidmi_m',
        'Day_Ahead_Price_EUR_MWh', 'Intra_Day_Price_EUR_MWh',
        'Ref_M1_kW', 'Ref_M2_kW',
        'Haselholz_Inflow_ls', 'Bidmi_Inflow_ls',
    ]

    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    numeric_cols = [
        'Forecast_kW', 'Consumption_kW', 'Haselholz_m', 'Bidmi_m',
        'Day_Ahead_Price_EUR_MWh', 'Intra_Day_Price_EUR_MWh',
        'Ref_M1_kW', 'Ref_M2_kW',
        'Haselholz_Inflow_ls', 'Bidmi_Inflow_ls',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    df['Bidmi_mm']     = df['Bidmi_m']     * 1000
    df['Haselholz_mm'] = df['Haselholz_m'] * 1000

    # Reference revenue evaluated on the same DA/ID pricing basis as all optimised modes.
    # DA bid = Forecast (same bid submitted regardless of mode).
    # ID cost = deviation of historical production from the forecast bid.
    # This makes the gain purely a measure of dispatch quality improvement,
    # independent of forecast accuracy.
    ref_prod = df['Ref_M2_kW'] + df['Ref_M1_kW']
    ref_da   = (df['Forecast_kW'] - df['Consumption_kW']) * TIMESTEP_HOURS * df['Day_Ahead_Price_EUR_MWh'] / 1000
    ref_id   = -(ref_prod - df['Forecast_kW']).abs() * TIMESTEP_HOURS * df['Intra_Day_Price_EUR_MWh'] / 1000
    df['Ref_Energy_Trading_EUR'] = ref_da + ref_id
    return df


# ---------------------------------------------------------------------------
# Seasonal target computation  (used by all modes)
# ---------------------------------------------------------------------------

def compute_seasonal_targets(df: pd.DataFrame) -> dict:
    """30-day centred rolling average of midnight reservoir levels.
    Returns dict: date -> (target_bidmi_mm, target_haselholz_mm)
    """
    midnight = df[df['DateTime'].dt.time == pd.Timestamp('00:00').time()].copy()
    midnight = midnight.sort_values('DateTime').set_index('DateTime')

    t_lb = midnight['Bidmi_mm'].rolling(ROLLING_WINDOW, center=True, min_periods=10).mean()
    t_lh = midnight['Haselholz_mm'].rolling(ROLLING_WINDOW, center=True, min_periods=10).mean()

    t_lb = t_lb.bfill().ffill()
    t_lh = t_lh.bfill().ffill().clip(upper=HASELHOLZ_LEVEL_MAX)

    return {
        dt.date(): (float(lb), float(lh))
        for dt, lb, lh in zip(t_lb.index, t_lb.values, t_lh.values)
    }


def compute_seasonal_production(df: pd.DataFrame) -> dict:
    """30-day rolling average of reference production at each 5-min time slot.

    Returns dict: date -> (avg_m2_array, avg_m1_array)
    Each array has one value per 5-min slot in the day (288 entries for a full day).
    """
    df_c = df.copy()
    df_c['date']     = df_c['DateTime'].dt.date
    df_c['time_idx'] = df_c.groupby('date').cumcount()

    pivot_m2 = df_c.pivot(index='date', columns='time_idx', values='Ref_M2_kW')
    pivot_m1 = df_c.pivot(index='date', columns='time_idx', values='Ref_M1_kW')

    avg_m2 = pivot_m2.rolling(ROLLING_WINDOW, center=True, min_periods=10).mean().bfill().ffill()
    avg_m1 = pivot_m1.rolling(ROLLING_WINDOW, center=True, min_periods=10).mean().bfill().ffill()

    avg_m2 = avg_m2.fillna(0.0).clip(lower=0.0, upper=P_MAX_M2)
    avg_m1 = avg_m1.fillna(0.0).clip(lower=0.0, upper=P_MAX_M1)

    return {
        date: (avg_m2.loc[date].values.tolist(), avg_m1.loc[date].values.tolist())
        for date in avg_m2.index
    }


def compute_intraday_targets(df: pd.DataFrame) -> dict:
    """30-day rolling average of reservoir level at each 5-min time slot.
    Returns dict: date -> (floor_b_array, floor_h_array)
    """
    df_c = df.copy()
    df_c['date']     = df_c['DateTime'].dt.date
    df_c['time_idx'] = df_c.groupby('date').cumcount()

    pivot_b = df_c.pivot(index='date', columns='time_idx', values='Bidmi_mm')
    pivot_h = df_c.pivot(index='date', columns='time_idx', values='Haselholz_mm')

    floor_b = pivot_b.rolling(ROLLING_WINDOW, center=True, min_periods=10).mean().bfill().ffill()
    floor_h = pivot_h.rolling(ROLLING_WINDOW, center=True, min_periods=10).mean().bfill().ffill()

    floor_b = floor_b.fillna(BIDMI_LEVEL_MIN).clip(upper=BIDMI_LEVEL_MAX)
    floor_h = floor_h.fillna(HASELHOLZ_LEVEL_MIN).clip(upper=HASELHOLZ_LEVEL_MAX)

    return {
        date: (floor_b.loc[date].values.tolist(), floor_h.loc[date].values.tolist())
        for date in floor_b.index
    }


# ---------------------------------------------------------------------------
# Water balance helper  (one 5-min step)
# ---------------------------------------------------------------------------

def water_balance_step(lb, lh, m2, m1, inflow_b, inflow_h):
    """Simulate one timestep. Returns (lb_new, lh_new, spill_b_mm, spill_h_mm)."""
    raw_lb  = lb + inflow_b / BIDMI_LS_PER_MM - COEFF_M2_BIDMI * m2
    spill_b = max(0.0, raw_lb - BIDMI_LEVEL_MAX)
    lb_new  = max(BIDMI_LEVEL_MIN, raw_lb - spill_b)

    raw_lh  = lh + inflow_h / HASELHOLZ_LS_PER_MM + COEFF_M2_CASCADE * m2 - COEFF_M1_HASELHOLZ * m1
    spill_h = max(0.0, raw_lh - HASELHOLZ_LEVEL_MAX)
    lh_new  = max(HASELHOLZ_LEVEL_MIN, raw_lh - spill_h)

    return lb_new, lh_new, spill_b, spill_h


def spill_to_kwh(spill_b_mm: list, spill_h_mm: list):
    """Convert per-step spill [mm] to energy loss [kWh]."""
    b_kwh = [s * BIDMI_LS_PER_MM * M2_KW_PER_LS * TIMESTEP_HOURS * (1 + TURBINE_CASCADE_RATIO)
             for s in spill_b_mm]
    h_kwh = [s * HASELHOLZ_LS_PER_MM * M1_KW_PER_LS * TIMESTEP_HOURS
             for s in spill_h_mm]
    return b_kwh, h_kwh


# ---------------------------------------------------------------------------
# Common result columns  (written by every dispatch_day function)
# ---------------------------------------------------------------------------

def attach_common_results(day_df, opt_m2, opt_m1, opt_lb, opt_lh,
                          opt_spill_b, opt_spill_h,
                          demand, target_lb, target_lh, mode_name):
    """Add all standard Opt_* columns to day_df in-place.

    NOTE: Opt_Energy_Trading_EUR is intentionally NOT set here.
    Each mode (FORECAST, WATER_VALUE, WATER_LEVEL) writes its own
    DA + ID formula immediately after calling this function.
    """
    N     = len(day_df)
    opt_p = [opt_m2[t] + opt_m1[t] for t in range(N)]

    spill_b_kwh, spill_h_kwh = spill_to_kwh(opt_spill_b, opt_spill_h)

    day_df['Opt_M2_kW']         = opt_m2
    day_df['Opt_M1_kW']         = opt_m1
    day_df['Opt_Production_kW'] = opt_p
    day_df['Opt_Bidmi_mm']      = opt_lb
    day_df['Opt_Haselholz_mm']  = opt_lh

    # Opt_Energy_Trading_EUR is NOT set here — each mode computes its own
    # DA+ID pricing formula and writes the column directly after this call.

    day_df['Opt_Target_Bidmi_mm']           = target_lb
    day_df['Opt_Target_Haselholz_mm']       = target_lh
    day_df['Opt_Terminal_Dev_Bidmi_mm']     = 0.0
    day_df['Opt_Terminal_Dev_Haselholz_mm'] = 0.0
    day_df.loc[day_df.index[-1], 'Opt_Terminal_Dev_Bidmi_mm']     = opt_lb[-1] - target_lb
    day_df.loc[day_df.index[-1], 'Opt_Terminal_Dev_Haselholz_mm'] = opt_lh[-1] - target_lh

    day_df['Opt_Spill_Bidmi_mm']          = opt_spill_b
    day_df['Opt_Spill_Haselholz_mm']      = opt_spill_h
    day_df['Opt_Spill_Bidmi_kWh']         = spill_b_kwh
    day_df['Opt_Spill_Haselholz_kWh']     = spill_h_kwh
    day_df['Opt_Spill_Bidmi_kWh_Cum']     = pd.Series(spill_b_kwh).cumsum().tolist()
    day_df['Opt_Spill_Haselholz_kWh_Cum'] = pd.Series(spill_h_kwh).cumsum().tolist()

    THRESHOLD_B = BIDMI_LEVEL_MIN     + 0.20 * BIDMI_RANGE
    THRESHOLD_H = HASELHOLZ_LEVEL_MIN + 0.20 * HASELHOLZ_RANGE
    day_df['Opt_Recovery_Bidmi']     = [1 if lb < THRESHOLD_B else 0 for lb in opt_lb]
    day_df['Opt_Recovery_Haselholz'] = [1 if lh < THRESHOLD_H else 0 for lh in opt_lh]
    day_df['Opt_Mode']               = mode_name

    # Carry end-of-day levels as attributes for day chaining
    day_df._opt_lb_end = opt_lb[-1]
    day_df._opt_lh_end = opt_lh[-1]
    day_df._dev_b      = opt_lb[-1] - target_lb
    day_df._dev_h      = opt_lh[-1] - target_lh

    return day_df
