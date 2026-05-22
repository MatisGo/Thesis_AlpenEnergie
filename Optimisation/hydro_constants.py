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
# Currency conversion  (CHF → EUR, 2025 average exchange rate)
# ---------------------------------------------------------------------------
# 1 CHF = EUR_PER_CHF EUR. Use this whenever a CHF-denominated value enters
# the model (only the grid peak tariff right now — energy prices are already
# in EUR/MWh natively).
EUR_PER_CHF = 0.9370   # 2025 average

# ---------------------------------------------------------------------------
# Grid exchange peak tariff  (Leistungsentgelt)
# ---------------------------------------------------------------------------
# Real Swiss tariff is quoted as 10 CHF/kW/month at the grid connection.
# Convert to EUR for the model's unified currency.
PEAK_TARIFF_EUR_KW = 10.0 * EUR_PER_CHF   # ≈ 9.37 EUR/kW/month

# Peak-tariff time window — only 15-min import blocks whose timestamps fall
# within [start, end) hours count toward the monthly peak. Night imports are
# free of the peak fee. Change here to widen/narrow the active window.
PEAK_TARIFF_HOUR_START = 7    # inclusive (07:00)
PEAK_TARIFF_HOUR_END   = 20   # exclusive (20:00)

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
# Production forecast scaling
# ---------------------------------------------------------------------------
FORECAST_SCALE = 1.00   # apply to Forecast_kW (DA bid production side)

# ---------------------------------------------------------------------------
# Data file
# ---------------------------------------------------------------------------
DATA_FILENAME = os.path.join('Input', 'matis_2025_.xlsx')

# Imbalance-forecast variants stored in Input/imbalance_forecasts.xlsx.
# Use the short tag in run_optimiser.py's IMBALANCE_FORECAST setting.
IMBALANCE_FORECASTS_PATH = os.path.join('Input', 'imbalance_forecasts.xlsx')
IMBALANCE_VARIANT_TAGS = {
    'PERFECT':      'Perfect',       # perfect foresight (uses realised BG prices)
    'ROLLING_MEAN': 'Rolling_Mean',  # 7-day rolling mean shifted 1 day (today's method)
    'R2_02':        'R2_20',         # synthetic AR(1) noise, R^2 = 0.2
    'R2_05':        'R2_50',         # synthetic AR(1) noise, R^2 = 0.5
    'R2_08':        'R2_80',         # synthetic AR(1) noise, R^2 = 0.8
}

# Consumption-forecast variants stored in Input/consumption_forecasts.xlsx.
# Use the short tag in run_optimiser.py's CONSUMPTION_FORECAST setting.
CONSUMPTION_FORECASTS_PATH = os.path.join('Input', 'consumption_forecasts.xlsx')
CONSUMPTION_VARIANT_TAGS = {
    'PERFECT':  'Perfect',     # zero-error (forecast = real)
    'CNN_LSTM': 'CNN_LSTM',    # existing CNN-LSTM forecast (WMAPE ~6%)
    'MAPE_10':  'MAPE_10',     # synthetic, hour-of-day-conditional, WMAPE ~10%
    'MAPE_15':  'MAPE_15',     # synthetic, WMAPE ~15%
    'MAPE_20':  'MAPE_20',     # synthetic, WMAPE ~20%
}

# DA-price-forecast variants stored in Input/da_price_forecasts.xlsx.
# Only affects the LP_OPTIMISED production forecast LP (production_forecast_lp.py).
# Settlement always uses the real Day_Ahead_Price_EUR_MWh column.
DA_PRICE_FORECASTS_PATH = os.path.join('Input', 'da_price_forecasts.xlsx')
DA_PRICE_VARIANT_TAGS = {
    'PERFECT': 'Perfect',   # zero-error (= real DA price, upper bound)
    'MAPE_05': 'MAPE_05',   # additive AR(1) noise, MAPE ~5%  (best-in-class)
    'MAPE_15': 'MAPE_15',   # additive AR(1) noise, MAPE ~15% (2022-volatile equivalent)
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str, imbalance_forecast: str = 'ROLLING_MEAN',
              consumption_forecast: str = 'CNN_LSTM',
              da_price_forecast: str = 'PERFECT') -> pd.DataFrame:
    """Read matis_2025_.xlsx and return a clean DataFrame.

    imbalance_forecast   : key in IMBALANCE_VARIANT_TAGS — selects which BG
                           forecast variant feeds BG_*_Forecast_EUR_MWh columns.
    consumption_forecast : key in CONSUMPTION_VARIANT_TAGS — selects which
                           consumption-forecast variant replaces the
                           Forecast_Consumption_kW column.
                           Defaults to 'CURRENT' (use the column already in
                           matis_2025_.xlsx).
    da_price_forecast    : key in DA_PRICE_VARIANT_TAGS — selects the DA price
                           variant exposed as Day_Ahead_Price_Forecast_EUR_MWh.
                           Used only by the LP_OPTIMISED production forecast LP.
                           Settlement always uses Day_Ahead_Price_EUR_MWh (real).
                           Defaults to 'PERFECT' (= real price, no error).
    """
    raw = pd.read_excel(path, header=None, skiprows=3, engine='calamine')

    # Column layout (0-indexed, after skiprows=3):
    #   A=0 DateTime  E=4 Forecast_kW  G=6 Consumption_kW
    #   J=9 Haselholz_m  K=10 Bidmi_m
    #   L=11 Day_Ahead_Price_EUR_MWh
    #   N=13 Ref_M1_kW  O=14 Ref_M2_kW
    #   R=17 Haselholz_Inflow_ls  S=18 Bidmi_Inflow_ls
    #   T=19 BG_Long_EUR_MWh  U=20 BG_Short_EUR_MWh  (added by add_swissgrid_prices.py)
    df = raw.iloc[:, [0, 4, 6, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21]].copy()
    df.columns = [
        'DateTime', 'Forecast_kW', 'Consumption_kW',
        'Haselholz_m', 'Bidmi_m',
        'Day_Ahead_Price_EUR_MWh',
        'Ref_M1_kW', 'Ref_M2_kW',
        'Haselholz_Inflow_ls', 'Bidmi_Inflow_ls',
        'BG_Long_EUR_MWh', 'BG_Short_EUR_MWh',
        'Forecast_Consumption_kW',
    ]

    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    numeric_cols = [
        'Forecast_kW', 'Consumption_kW', 'Haselholz_m', 'Bidmi_m',
        'Day_Ahead_Price_EUR_MWh',
        'Ref_M1_kW', 'Ref_M2_kW',
        'Haselholz_Inflow_ls', 'Bidmi_Inflow_ls',
        'BG_Long_EUR_MWh', 'BG_Short_EUR_MWh',
        'Forecast_Consumption_kW',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    df['Bidmi_mm']     = df['Bidmi_m']     * 1000
    df['Haselholz_mm'] = df['Haselholz_m'] * 1000

    df['Forecast_kW'] *= FORECAST_SCALE

    # Consumption-forecast variant — pulled from Input/consumption_forecasts.xlsx
    # if available. 'CURRENT' = the column already loaded (no swap).
    cons_variant_tag = CONSUMPTION_VARIANT_TAGS.get(consumption_forecast)
    if cons_variant_tag is None:
        raise ValueError(f"Unknown consumption_forecast '{consumption_forecast}'. "
                         f"Choices: {list(CONSUMPTION_VARIANT_TAGS.keys())}")
    if consumption_forecast != 'CNN_LSTM':
        _cons_path = os.path.join(os.path.dirname(path), '..', 'Input', 'consumption_forecasts.xlsx')
        _cons_path = os.path.normpath(_cons_path)
        if not os.path.exists(_cons_path):
            _cons_path = os.path.join(os.path.dirname(path), 'consumption_forecasts.xlsx')
        if not os.path.exists(_cons_path):
            raise FileNotFoundError(
                f"consumption_forecasts.xlsx not found and variant "
                f"'{consumption_forecast}' requires it. "
                f"Run generate_consumption_forecasts.py first.")
        cfc = pd.read_excel(_cons_path, sheet_name='Forecasts',
                             parse_dates=['DateTime'], engine='calamine',
                             usecols=['DateTime',
                                      f'Forecast_Consumption_{cons_variant_tag}_kW'])
        cfc = cfc.set_index('DateTime').reindex(df['DateTime']).reset_index(drop=True)
        df['Forecast_Consumption_kW'] = cfc[f'Forecast_Consumption_{cons_variant_tag}_kW'].values

    # Imbalance-price forecast columns — pulled from Input/imbalance_forecasts.xlsx
    # if available (allows comparing forecast-quality variants), otherwise fall
    # back to the legacy 7-day rolling mean shifted 1 day forward (no look-ahead).
    _imb_path = os.path.join(os.path.dirname(path), '..', 'Input', 'imbalance_forecasts.xlsx')
    _imb_path = os.path.normpath(_imb_path)
    # Always-available Rolling-Mean BG forecast (used by the day-ahead production
    # forecast LP regardless of IMBALANCE_FORECAST selection). 7-day rolling mean
    # shifted 1 day forward — no look-ahead, matches the operational forecast.
    df['BG_Long_RollingMean_EUR_MWh']  = (df['BG_Long_EUR_MWh']
                                            .rolling(2016, min_periods=48).mean()
                                            .shift(288).bfill().ffill())
    df['BG_Short_RollingMean_EUR_MWh'] = (df['BG_Short_EUR_MWh']
                                            .rolling(2016, min_periods=48).mean()
                                            .shift(288).bfill().ffill())

    if not os.path.exists(_imb_path):
        # Try same folder as matis_2025_.xlsx
        _imb_path = os.path.join(os.path.dirname(path), 'imbalance_forecasts.xlsx')

    variant_tag = IMBALANCE_VARIANT_TAGS.get(imbalance_forecast)
    if variant_tag is None:
        raise ValueError(f"Unknown imbalance_forecast '{imbalance_forecast}'. "
                         f"Choices: {list(IMBALANCE_VARIANT_TAGS.keys())}")

    if os.path.exists(_imb_path):
        fc = pd.read_excel(_imb_path, sheet_name='Forecasts',
                            parse_dates=['DateTime'], engine='calamine',
                            usecols=['DateTime',
                                     f'BG_Long_{variant_tag}_EUR_MWh',
                                     f'BG_Short_{variant_tag}_EUR_MWh'])
        # Align on DateTime
        fc = fc.set_index('DateTime').reindex(df['DateTime']).reset_index(drop=True)
        df['BG_Long_Forecast_EUR_MWh']  = fc[f'BG_Long_{variant_tag}_EUR_MWh'].values
        df['BG_Short_Forecast_EUR_MWh'] = fc[f'BG_Short_{variant_tag}_EUR_MWh'].values
    else:
        # Fallback: compute the legacy 7-day rolling mean (ROLLING_MEAN variant only)
        if imbalance_forecast != 'ROLLING_MEAN':
            raise FileNotFoundError(
                f"imbalance_forecasts.xlsx not found and "
                f"variant '{imbalance_forecast}' requires it. "
                f"Run generate_imbalance_forecasts.py first.")
        df['BG_Long_Forecast_EUR_MWh']  = (df['BG_Long_EUR_MWh']
                                            .rolling(2016, min_periods=48).mean()
                                            .shift(288).bfill().ffill())
        df['BG_Short_Forecast_EUR_MWh'] = (df['BG_Short_EUR_MWh']
                                            .rolling(2016, min_periods=48).mean()
                                            .shift(288).bfill().ffill())

    # Reference revenue: DA bid = Forecast_Production - Forecast_Consumption (net position).
    # Imbalance = (Actual_Net - DA_bid) settled at real BG prices (asymmetric).
    # dev > 0 (long/over-delivery) → BG-long price; dev < 0 (short) → BG-short price.
    ref_prod   = df['Ref_M2_kW'] + df['Ref_M1_kW']
    da_bid     = df['Forecast_kW'] - df['Forecast_Consumption_kW']
    ref_da     = da_bid * TIMESTEP_HOURS * df['Day_Ahead_Price_EUR_MWh'] / 1000
    actual_net = ref_prod - df['Consumption_kW']
    dev        = actual_net - da_bid
    ref_id_long  = dev.clip(lower=0) * df['BG_Long_EUR_MWh']  * TIMESTEP_HOURS / 1000
    ref_id_short = dev.clip(upper=0) * df['BG_Short_EUR_MWh'] * TIMESTEP_HOURS / 1000
    ref_id       = ref_id_long + ref_id_short
    df['Ref_DA_EUR']              = ref_da
    df['Ref_Imbalance_Long_EUR']  = ref_id_long
    df['Ref_Imbalance_Short_EUR'] = ref_id_short
    df['Ref_Imbalance_EUR']       = ref_id
    df['Ref_Energy_Trading_EUR']  = ref_da + ref_id

    # DA-price forecast variant — exposed as Day_Ahead_Price_Forecast_EUR_MWh.
    # Used by production_forecast_lp.py; settlement always uses the real column.
    da_fc_tag = DA_PRICE_VARIANT_TAGS.get(da_price_forecast)
    if da_fc_tag is None:
        raise ValueError(f"Unknown da_price_forecast '{da_price_forecast}'. "
                         f"Choices: {list(DA_PRICE_VARIANT_TAGS.keys())}")
    if da_price_forecast == 'PERFECT':
        df['Day_Ahead_Price_Forecast_EUR_MWh'] = df['Day_Ahead_Price_EUR_MWh']
    else:
        _da_path = os.path.join(os.path.dirname(path), '..', 'Input', 'da_price_forecasts.xlsx')
        _da_path = os.path.normpath(_da_path)
        if not os.path.exists(_da_path):
            _da_path = os.path.join(os.path.dirname(path), 'da_price_forecasts.xlsx')
        if not os.path.exists(_da_path):
            raise FileNotFoundError(
                f"da_price_forecasts.xlsx not found and variant "
                f"'{da_price_forecast}' requires it. "
                f"Run generate_da_price_forecasts.py first.")
        dfc = pd.read_excel(_da_path, sheet_name='Forecasts',
                            parse_dates=['DateTime'], engine='calamine',
                            usecols=['DateTime', f'DA_Price_{da_fc_tag}_EUR_MWh'])
        dfc = dfc.set_index('DateTime').reindex(df['DateTime']).reset_index(drop=True)
        df['Day_Ahead_Price_Forecast_EUR_MWh'] = (
            dfc[f'DA_Price_{da_fc_tag}_EUR_MWh']
              .interpolate('linear').bfill().ffill()
              .values
        )

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
    """Simulate one 5-min timestep, enforcing physical reservoir limits.

    Each turbine is clamped down to whatever the reservoir can actually supply
    without dropping below its minimum level — silently clamping just the
    level (as the previous version did) credited phantom production.

    Returns (lb_new, lh_new, spill_b_mm, spill_h_mm, m2_actual, m1_actual).
    """
    # M2 cannot drain Bidmi below BIDMI_LEVEL_MIN this step
    m2_max = (lb + inflow_b / BIDMI_LS_PER_MM - BIDMI_LEVEL_MIN) / COEFF_M2_BIDMI
    m2     = max(0.0, min(m2, m2_max))

    raw_lb  = lb + inflow_b / BIDMI_LS_PER_MM - COEFF_M2_BIDMI * m2
    spill_b = max(0.0, raw_lb - BIDMI_LEVEL_MAX)
    lb_new  = max(BIDMI_LEVEL_MIN, raw_lb - spill_b)

    # M1 cannot drain Haselholz below HASELHOLZ_LEVEL_MIN, given the chosen m2 cascade
    hh_after_m2 = lh + inflow_h / HASELHOLZ_LS_PER_MM + COEFF_M2_CASCADE * m2
    m1_max      = (hh_after_m2 - HASELHOLZ_LEVEL_MIN) / COEFF_M1_HASELHOLZ
    m1          = max(0.0, min(m1, m1_max))

    raw_lh  = hh_after_m2 - COEFF_M1_HASELHOLZ * m1
    spill_h = max(0.0, raw_lh - HASELHOLZ_LEVEL_MAX)
    lh_new  = max(HASELHOLZ_LEVEL_MIN, raw_lh - spill_h)

    return lb_new, lh_new, spill_b, spill_h, m2, m1


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
    Each mode (RULE_BASED_DISPATCH, OPTIMISATION_DISPATCH) writes its own
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
