"""
CNN-LSTM 48h Load Prediction (Net Load with PV Conditioning)
=============================================================
Hybrid CNN-LSTM model for 48-hour ahead net electricity load forecasting.

Net load = grid consumption = gross load - local PV production.
The midday "duck curve" dip caused by rooftop/local PV is modelled via
an estimated PV production series fed as a third decoder input.

Architecture (based on Chung & Jang 2022, Khan et al. 2020):
  Encoder:
    - 3x Conv1D blocks with aggressive pooling (2016 -> 63 timesteps)
    - 2x LSTM layers for temporal pattern learning
  Decoder conditioning (3 inputs merged):
    - Input 2: Same-weekday statistical profiles (past 4 weeks, 576×4)
    - Input 3: Estimated PV production for forecast horizon (576×1)
      Built via Lorenz (2011) / Bacher (2009) correction table:
        PV_Est = Irr_FC × ratio[month, 15min_slot]
      ratio table is pre-computed from 2 years of PV production data.
  Decoder:
    - RepeatVector + LSTM decoder + TimeDistributed Dense
    - Outputs (48 hours, 12 steps each) then flattened to 576 steps

Input 1 - Encoder (7-day lookback, 2016 steps x 11 features):
  Load_Is (past), Load_yesterday, Load_last_week,
  Hour_sin, Hour_cos, Weekday_sin, Weekday_cos,
  PHolyday, Temp_Forecast, Rain_Forecast,
  Irr_FC    ← API irradiance forecast (GHI from Open-Meteo, consistent source for encoder)

Input 2 - Decoder conditioning (576 steps x 4 weekly profiles):
  Load from same weekday 1w, 2w, 3w, 4w ago for each forecast step

Input 3 - PV estimate for forecast horizon (576 steps x 1):
  Estimated PV production (kW) using irradiance forecast × correction table

Output: Load_Is for the next 48 hours (576 steps at 5-min resolution)
Loss:   Huber (less sensitive to spikes than MSE)

References:
  Lorenz et al. (2011) DOI: 10.1002/pip.1033
  Bacher et al. (2009) DOI: 10.1016/j.solener.2009.05.016

Usage:
  python CNN_LSTM_Prediction.py --train
  python CNN_LSTM_Prediction.py --predict 2026-02-15
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, BatchNormalization, Input,
                                      RepeatVector, TimeDistributed, Reshape,
                                      Flatten, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(SCRIPT_DIR, 'Simulation results')
DATA_PATH     = os.path.join(SCRIPT_DIR, '..', 'Data_Prediction.xlsx')
WEATHER_PATH  = os.path.join(SCRIPT_DIR, 'Imported_Forecast.xlsx')
MODEL_PATH    = os.path.join(SCRIPT_DIR, 'Model', 'CNN_LSTM_Model.keras')
CONFIG_PATH   = os.path.join(SCRIPT_DIR, 'Model', 'CNN_LSTM_Config.npz')
PV_TABLE_PATH = os.path.join(SCRIPT_DIR, 'PV_Correction', 'PV_Correction_Table.npz')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Data parameters
LOOKBACK_STEPS = 2016       # 7 days * 288 steps/day (5-min resolution)
FORECAST_HORIZON = 576      # 2 days * 288 steps/day
STEP_SIZE = 6               # Sample every 30 min (reduces memory, keeps enough samples)
METADATA_ROWS = 2           # Rows to skip in xlsx (Einheit, Signalname)
TEST_DAYS = 10              # Last N days held out for testing

# Input features (order matters - must match during prediction)
INPUT_FEATURES = [
    'Load_Is',              # Past load values (net load = gross - PV)
    'Load_yesterday',       # Load at same time 24h ago (lagged feature)
    'Load_last_week',       # Load at same time 7 days ago (lagged feature)
    'Hour_sin',             # Time of day (sin component)
    'Hour_cos',             # Time of day (cos component)
    'Weekday_sin',          # Day of week (sin component)
    'Weekday_cos',          # Day of week (cos component)
    'PHolyday',             # Public holiday flag (0/1)
    'Temp_Forecast',        # Temperature forecast (deg C)
    'Rain_Forecast',        # Rain forecast (mm)
    'Irr_FC',               # API irradiance forecast (W/m²) - GHI from Open-Meteo, solar context for encoder
]
TARGET = 'Load_Is'

# Output reshaping: 48 hourly blocks x 12 steps (5-min) per block = 576
OUTPUT_HOURS = 48
STEPS_PER_HOUR = 12

# CNN architecture
CNN_FILTERS = [64, 128, 256]
CNN_KERNELS = [5, 5, 3]
POOL_SIZES = [4, 4, 2]     # 2016 → 504 → 126 → 63

# LSTM architecture
LSTM_UNITS = [128, 64]

# Dense output
DENSE_UNITS = 256
DROPOUT_RATE = 0.2

# Training
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15

# Statistical baseline
HYBRID_N_PAST_WEEKS = 4        # Number of same-day-of-week weeks to look back
HYBRID_AGG_METHOD = 'mean'     # 'mean' or 'median' for historical aggregation

# Decoder conditioning (same-weekday statistical profiles as second model input)
STAT_N_WEEKS = 4               # Number of past same-weekday profiles fed to decoder


# =============================================================================
# PV CORRECTION TABLE
# =============================================================================

def load_pv_correction_table():
    """
    Load the pre-computed irradiance-to-PV correction table.

    The table was built by Build_PV_Correction_Table.py from 2 years of
    measured PV production and co-located irradiance forecast data,
    following the Lorenz (2011) / Bacher (2009) approach.

    Returns
    -------
    ratio_table : np.ndarray, shape (12, 96)
        ratio_table[month-1, slot] = kW per (W/m²)
        where slot is the 15-min slot index (0 = 00:00, 95 = 23:45).
        Returns None if the file does not exist (PV feature disabled).
    """
    if not os.path.exists(PV_TABLE_PATH):
        print(f"WARNING: PV correction table not found at {PV_TABLE_PATH}")
        print("  Run Build_PV_Correction_Table.py first to enable PV conditioning.")
        print("  Training will proceed WITHOUT PV input (Irr_FC still in encoder).")
        return None

    data = np.load(PV_TABLE_PATH)
    ratio_table = data['ratio_table'].astype(np.float32)  # (12, 96)
    print(f"  PV correction table loaded: shape={ratio_table.shape}, "
          f"max_ratio={ratio_table.max():.4f}")
    return ratio_table


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def _load_weather(path: str) -> pd.DataFrame:
    """
    Load Imported_Forecast.xlsx (produced by get_weather_data.py).

    Returns DataFrame with columns: [DateTime, Temperature_C, Irradiance_Wm2, Rain_Sum_mm]
    """
    df_w = pd.read_excel(path, sheet_name='Weather_Data')
    df_w['DateTime'] = pd.to_datetime(df_w['Time'], format='%d.%m.%Y %H:%M:%S',
                                       errors='coerce')
    df_w = df_w.dropna(subset=['DateTime'])
    df_w['Temperature_C']  = pd.to_numeric(df_w['Temperature_C'],  errors='coerce')
    df_w['Irradiance_Wm2'] = pd.to_numeric(df_w['Irradiance_Wm2'], errors='coerce').clip(lower=0)
    df_w['Rain_Sum_mm']    = pd.to_numeric(df_w['Rain_Sum_mm'],    errors='coerce').fillna(0.0)
    return df_w[['DateTime', 'Temperature_C', 'Irradiance_Wm2', 'Rain_Sum_mm']]


def load_data(ratio_table=None):
    """
    Load and preprocess data from two sources:
      1. Data_Prediction.xlsx  — historical load, real sensor values (Irr_Real),
                                 Forecast_Load (graph only), calendar features.
      2. Imported_Forecast.xlsx — Open-Meteo API: Temperature, Irradiance (Irr_FC),
                                   Rain. Covers historical + next 72h forecast.

    Merge strategy (outer join on DateTime):
      - Irr_Real      → real sensor 'Irradiance Meiringen' from Data_Prediction (kept for reference)
      - Temp_Forecast → Temperature_C from Imported_Forecast (API replaces old forecast)
      - Rain_Forecast → Rain_Sum_mm from Imported_Forecast (API replaces old forecast)
      - Irr_FC        → Irradiance_Wm2 from Imported_Forecast (API source for BOTH encoder feature
                         and PV_Est computation — single consistent irradiance source)
      - Future rows (beyond Data_Prediction) → calendar features computed from DateTime,
        Load_Is = NaN (enables 48h prediction beyond data end).

    Also computes PV_Est = Irr_FC × ratio_table[month, slot] if table is provided.
    """
    # -------------------------------------------------------------------------
    # 1. Load Data_Prediction.xlsx (historical: load, real sensors, calendar)
    # -------------------------------------------------------------------------
    print("Loading main data from:", DATA_PATH)
    df = pd.read_excel(DATA_PATH, header=0)
    df = df.iloc[METADATA_ROWS:].reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]

    # Parse datetime from Date (col B) + Day_Time (col C)
    date_part = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
    time_part = pd.to_timedelta(df['Day_Time'].astype(str), errors='coerce')
    df['DateTime'] = date_part + time_part

    # Numeric columns from main file
    for col in ['Load_Is', 'Forecast_Load', 'Weekday', 'PHolyday']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Real irradiance sensor (kept from Data_Prediction — not replaced by API)
    irr_real_col = 'Irradiance Meiringen'
    if irr_real_col in df.columns:
        df['Irr_Real'] = pd.to_numeric(df[irr_real_col], errors='coerce').clip(lower=0)
    else:
        print(f"  WARNING: '{irr_real_col}' not found — Irr_Real set to 0")
        df['Irr_Real'] = 0.0

    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 2. Load Imported_Forecast.xlsx (API: Temperature, Irradiance FC, Rain)
    # -------------------------------------------------------------------------
    if os.path.exists(WEATHER_PATH):
        print("Loading weather data from:", WEATHER_PATH)
        df_w = _load_weather(WEATHER_PATH)

        # Outer merge: keeps all rows from both sources
        # Future rows (only in df_w) will have NaN for load/grid columns
        df = df.merge(df_w, on='DateTime', how='outer').sort_values('DateTime').reset_index(drop=True)
        print(f"  Weather merge: {len(df_w)} API rows merged — "
              f"future rows added: {(df['Load_Is'].isna() & df['Temperature_C'].notna()).sum()}")
    else:
        print(f"  WARNING: '{WEATHER_PATH}' not found — using Data_Prediction weather columns")
        df['Temperature_C']  = pd.to_numeric(df.get('Temp_Forecast',  0), errors='coerce')
        df['Irradiance_Wm2'] = pd.to_numeric(df.get('Irradiance Forecast', 0), errors='coerce').clip(lower=0)
        df['Rain_Sum_mm']    = pd.to_numeric(df.get('Rain_Forecast',  0), errors='coerce').fillna(0)

    # -------------------------------------------------------------------------
    # 3. Compute / fill calendar features for ALL rows (incl. future API rows)
    # -------------------------------------------------------------------------
    # Hour cyclical encoding — computed from DateTime directly (always valid)
    df['HourFrac'] = df['DateTime'].dt.hour + df['DateTime'].dt.minute / 60.0
    df['Hour_sin'] = np.sin(2 * np.pi * df['HourFrac'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['HourFrac'] / 24)

    # Weekday: fill NaN (future rows have no Weekday from Data_Prediction)
    df['Weekday'] = df['Weekday'].fillna(df['DateTime'].dt.weekday.astype(float))
    df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

    # PHolyday: default 0 for future rows (unknown)
    df['PHolyday'] = df['PHolyday'].fillna(0)

    # -------------------------------------------------------------------------
    # 4. Map API weather columns → model feature names
    # -------------------------------------------------------------------------
    # Temp_Forecast: API temperature (replaces old 'Temp_Forecast' from Data_Prediction)
    df['Temp_Forecast'] = df['Temperature_C'].fillna(
        pd.to_numeric(df.get('Temp_Forecast', pd.Series(dtype=float)), errors='coerce'))

    # Rain_Forecast: API daily rain sum (replaces old 'Rain_Forecast')
    df['Rain_Forecast'] = df['Rain_Sum_mm'].fillna(0.0)

    # Irr_FC: API irradiance forecast (replaces old 'Irradiance Forecast')
    df['Irr_FC'] = df['Irradiance_Wm2'].fillna(0.0).clip(lower=0)

    # Irr_Real: real sensor from Data_Prediction for historical rows
    # For future rows (no sensor), use API irradiance as proxy
    df['Irr_Real'] = df['Irr_Real'].fillna(df['Irradiance_Wm2']).fillna(0.0).clip(lower=0)

    # -------------------------------------------------------------------------
    # 5. Interpolate continuous features and compute lagged load
    # -------------------------------------------------------------------------
    for col in ['Temp_Forecast', 'Rain_Forecast']:
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # Load_Is: only interpolate within historical region (not into future NaN)
    df['Load_Is'] = df['Load_Is'].interpolate(method='linear', limit_direction='both',
                                               limit_area='inside')

    df['Load_yesterday'] = df['Load_Is'].shift(288).bfill()
    df['Load_last_week'] = df['Load_Is'].shift(2016).bfill()

    # -------------------------------------------------------------------------
    # 6. PV estimate via Lorenz (2011) correction table
    # -------------------------------------------------------------------------
    if ratio_table is not None:
        months = df['DateTime'].dt.month.values - 1
        slots  = (df['DateTime'].dt.hour * 4 +
                  df['DateTime'].dt.minute // 15).values
        slots  = np.clip(slots, 0, 95)
        ratio_per_step = ratio_table[months, slots]
        df['PV_Est'] = (df['Irr_FC'].values * ratio_per_step).astype(np.float32)
    else:
        df['PV_Est'] = 0.0

    print(f"  Total rows     : {len(df)}")
    print(f"  Date range     : {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"  Rows with Load : {df['Load_Is'].notna().sum()}")
    print(f"  Future rows    : {df['Load_Is'].isna().sum()}")
    print(f"  PV_Est range   : [{df['PV_Est'].min():.1f}, {df['PV_Est'].max():.1f}] kW")

    return df


# =============================================================================
# SEQUENCE CREATION
# =============================================================================

def create_sequences(df, step=STEP_SIZE):
    """
    Create sliding-window input/output sequences.

    Each sample:
      X       = 7 days of features BEFORE the prediction point
      y       = 48 hours of Load_Is AFTER the prediction point
      pv_horiz= PV_Est for the 48-hour forecast window (decoder input 3)

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data (must have INPUT_FEATURES, TARGET, DateTime,
        Forecast_Load, PV_Est).
    step : int
        Sliding window step size (in 5-min increments).

    Returns
    -------
    X, y, timestamps, forecast_loads, start_indices, pv_horizons : np.ndarray
    """
    features = df[INPUT_FEATURES].values
    targets = df[TARGET].values
    timestamps = df['DateTime'].values
    forecast_loads = pd.to_numeric(df['Forecast_Load'], errors='coerce').values
    pv_est_values = df['PV_Est'].values.astype(np.float32)

    X, y, ts, fc, idx_list, pv_list = [], [], [], [], [], []

    for i in range(LOOKBACK_STEPS, len(df) - FORECAST_HORIZON + 1, step):
        target_slice = targets[i:i + FORECAST_HORIZON]
        input_slice = features[i - LOOKBACK_STEPS:i]

        # Skip samples with NaN in input or target
        if np.isnan(target_slice).any() or np.isnan(input_slice).any():
            continue

        X.append(input_slice)
        y.append(target_slice)
        ts.append(timestamps[i])
        fc.append(forecast_loads[i:i + FORECAST_HORIZON])
        idx_list.append(i)  # forecast start index (for stat profiles)
        pv_list.append(pv_est_values[i:i + FORECAST_HORIZON])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    ts = np.array(ts)
    fc = np.array(fc, dtype=np.float32)
    start_indices = np.array(idx_list, dtype=np.int64)
    pv_horizons = np.array(pv_list, dtype=np.float32)  # (N, 576)

    print(f"  Created {len(X)} sequences")
    print(f"  X shape: {X.shape}  (samples, lookback, features)")
    print(f"  y shape: {y.shape}  (samples, forecast_horizon)")
    print(f"  PV horizons shape: {pv_horizons.shape}")

    return X, y, ts, fc, start_indices, pv_horizons


# =============================================================================
# STATISTICAL PROFILES FOR DECODER CONDITIONING
# =============================================================================

def build_stat_profiles(load_values, start_indices, horizon=FORECAST_HORIZON,
                        n_weeks=STAT_N_WEEKS):
    """
    Build same-weekday statistical profiles for the forecast horizon.

    For each sample starting at index i, extract the Load_Is values from
    N past same-weekdays (1w, 2w, 3w, 4w ago) over the forecast window.
    This gives the decoder direct visibility into weekly recurring patterns.

    Parameters
    ----------
    load_values : np.ndarray
        Full Load_Is column from the dataset.
    start_indices : np.ndarray of int
        Forecast start index for each sample (same as in create_sequences).
    horizon : int
        Forecast horizon in steps (576).
    n_weeks : int
        Number of past same-weekday profiles to include.

    Returns
    -------
    profiles : np.ndarray, shape (n_samples, horizon, n_weeks)
        Load values from past same-weekdays for each forecast step.
    """
    n = len(start_indices)
    profiles = np.full((n, horizon, n_weeks), np.nan, dtype=np.float32)

    for w in range(1, n_weeks + 1):
        shift = w * 2016  # w weeks back at 5-min resolution
        for s, i in enumerate(start_indices):
            src_start = i - shift
            src_end = src_start + horizon
            if src_start >= 0 and src_end <= len(load_values):
                profiles[s, :, w - 1] = load_values[src_start:src_end]
            elif src_start >= 0:
                valid_len = min(len(load_values) - src_start, horizon)
                profiles[s, :valid_len, w - 1] = load_values[src_start:src_start + valid_len]

    # Fill NaN: for each sample/step, use the mean of available weeks
    for s in range(n):
        for j in range(horizon):
            row = profiles[s, j, :]
            available = row[np.isfinite(row)]
            if len(available) > 0 and np.isnan(row).any():
                profiles[s, j, np.isnan(row)] = available.mean()
            elif len(available) == 0:
                # No history at all → use nearest valid value in same sample
                profiles[s, j, :] = 0.0  # will be rare; scaler handles it

    return profiles


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_model(input_shape, stat_shape, pv_shape, output_hours=OUTPUT_HOURS,
                steps_per_hour=STEPS_PER_HOUR):
    """
    Build CNN-LSTM model with triple decoder conditioning.

    Triple-input architecture:
      Input 1 (encoder): 7-day lookback features → CNN → LSTM → encoded context
      Input 2 (decoder cond.): Same-weekday load profiles for forecast horizon
      Input 3 (PV estimate): PV_Est for forecast horizon (Lorenz 2011 approach)

    The decoder receives the encoded context, the weekly load pattern, AND
    the estimated PV production, allowing it to predict the midday net-load
    dip caused by local solar generation (the "duck curve" effect).

    Parameters
    ----------
    input_shape : tuple
        Encoder input: (timesteps, features) = (2016, 11)
    stat_shape : tuple
        Statistical profile: (forecast_horizon, n_weeks) = (576, 4)
    pv_shape : tuple
        PV estimate: (forecast_horizon, 1) = (576, 1)
    output_hours : int
        Number of hourly blocks = 48
    steps_per_hour : int
        5-min steps per hour = 12

    Returns
    -------
    model : keras.Model
    """
    # --- Three inputs ---
    encoder_input = Input(shape=input_shape, name='encoder_input')
    stat_input    = Input(shape=stat_shape,  name='stat_input')
    pv_input      = Input(shape=pv_shape,    name='pv_input')

    # --- CNN Encoder (3 blocks: 2016 -> 504 -> 126 -> 63) ---
    x = encoder_input
    for filters, kernel, pool in zip(CNN_FILTERS, CNN_KERNELS, POOL_SIZES):
        x = Conv1D(filters, kernel, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=pool)(x)
        x = Dropout(DROPOUT_RATE)(x)

    # --- LSTM Encoder ---
    x = LSTM(LSTM_UNITS[0], return_sequences=True)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = LSTM(LSTM_UNITS[1], return_sequences=False)(x)
    x = Dropout(DROPOUT_RATE)(x)

    # --- Decoder: produce structured output (48 hours x 12 steps) ---
    # Encode context into dense vector, repeat for each output hour
    x = Dense(DENSE_UNITS, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = RepeatVector(output_hours)(x)  # (batch, 48, 256)

    # --- Process statistical profiles (Input 2) ---
    # Reshape (576, n_weeks) → (48, 12 * n_weeks) to match hourly decoder blocks
    n_weeks = stat_shape[1]
    stat = Reshape((output_hours, steps_per_hour * n_weeks))(stat_input)  # (batch, 48, 48)
    stat = TimeDistributed(Dense(32, activation='relu'))(stat)            # (batch, 48, 32)

    # --- Process PV estimate (Input 3) ---
    # Reshape (576, 1) → (48, 12) hourly blocks, then compress to 8-dim
    pv = Reshape((output_hours, steps_per_hour))(pv_input)               # (batch, 48, 12)
    pv = TimeDistributed(Dense(8, activation='relu'))(pv)                # (batch, 48, 8)

    # --- Concatenate encoder context + statistical + PV conditioning ---
    x = Concatenate()([x, stat, pv])  # (batch, 48, 256 + 32 + 8 = 296)

    # LSTM decoder processes each hour with context, weekly pattern, and PV
    x = LSTM(128, return_sequences=True)(x)  # (batch, 48, 128)
    x = Dropout(DROPOUT_RATE)(x)

    # TimeDistributed Dense: each hour -> 12 five-min steps
    x = TimeDistributed(Dense(64, activation='relu'))(x)  # (batch, 48, 64)
    x = TimeDistributed(Dense(steps_per_hour))(x)          # (batch, 48, 12)

    # Flatten to (batch, 576)
    outputs = Reshape((output_hours * steps_per_hour,))(x)

    model = Model(inputs=[encoder_input, stat_input, pv_input], outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=Huber(delta=1.0),  # Less sensitive to spikes than MSE
        metrics=['mae']
    )

    return model


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(actual, predicted):
    """Compute RMSE, MAE, MAPE between actual and predicted arrays."""
    actual = actual.flatten()
    predicted = predicted.flatten()

    # Filter out invalid values
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if valid.sum() == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}

    a = actual[valid]
    p = predicted[valid]

    rmse = np.sqrt(mean_squared_error(a, p))
    mae = mean_absolute_error(a, p)
    mape = np.mean(np.abs((a - p) / a)) * 100

    return {'rmse': rmse, 'mae': mae, 'mape': mape}


# =============================================================================
# STATISTICAL BASELINE
# =============================================================================

def compute_stat_baseline(df, forecast_timestamps,
                           n_past_weeks=HYBRID_N_PAST_WEEKS,
                           agg_method=HYBRID_AGG_METHOD):
    """
    Compute statistical same-day-of-week baseline for the forecast horizon.

    For each 5-min slot in the forecast, find the same weekday+time slot
    for the last n_past_weeks weeks and aggregate (mean or median).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DateTime and Load_Is columns.
    forecast_timestamps : list of pd.Timestamp
        Timestamps for each forecast step.
    n_past_weeks : int
        Number of same-day-of-week weeks to look back.
    agg_method : str
        'mean' or 'median' for aggregating historical values.

    Returns
    -------
    stat_values : np.ndarray
        Statistical baseline prediction, shape (FORECAST_HORIZON,).
    """
    n_steps = len(forecast_timestamps)
    stat_values = np.full(n_steps, np.nan)

    # Build DatetimeIndex-based Series for fast nearest-neighbour lookup
    df_load = df.set_index('DateTime')['Load_Is'].copy()
    df_load = df_load[~df_load.index.duplicated(keep='first')].sort_index()

    for i, ts in enumerate(forecast_timestamps):
        ts = pd.Timestamp(ts)
        historical_values = []

        for w in range(1, n_past_weeks + 1):
            hist_ts = ts - pd.Timedelta(weeks=w)
            idx_arr = df_load.index.get_indexer([hist_ts], method='nearest',
                                                 tolerance=pd.Timedelta(minutes=5))
            if idx_arr[0] >= 0:
                val = df_load.iloc[idx_arr[0]]
                if np.isfinite(val) and val > 0:
                    historical_values.append(val)

        if len(historical_values) >= 1:
            hist_arr = np.array(historical_values)
            if agg_method == 'median':
                stat_values[i] = np.median(hist_arr)
            else:
                stat_values[i] = np.mean(hist_arr)

    return stat_values


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_evaluation(y_test, y_pred, fc_test, ts_test, history):
    """
    Create evaluation plots comparing CNN-LSTM vs External Forecast.

    Plots:
      1. 48h forecast: 3 curves (Actual, CNN-LSTM, External)
      2. KPI bar chart comparison (RMSE, MAE, MAPE)
      3. Training history (loss curves)
      4. Actual vs Predicted scatter plot
    """
    # --- Compute KPIs across ALL test samples ---
    cnn_metrics = compute_metrics(y_test, y_pred)

    # External forecast: filter valid values (> 0)
    fc_flat = fc_test.flatten()
    y_flat = y_test.flatten()
    fc_valid = np.isfinite(fc_flat) & (fc_flat > 0) & np.isfinite(y_flat) & (y_flat > 0)
    if fc_valid.any():
        ext_metrics = compute_metrics(y_flat[fc_valid], fc_flat[fc_valid])
    else:
        ext_metrics = {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}

    # --- Print KPIs ---
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (across all test samples)")
    print("=" * 60)
    print(f"{'Metric':<10} {'CNN-LSTM':>12} {'External':>12}")
    print("-" * 36)
    print(f"{'RMSE':<10} {cnn_metrics['rmse']:>12.2f} {ext_metrics['rmse']:>12.2f}")
    print(f"{'MAE':<10} {cnn_metrics['mae']:>12.2f} {ext_metrics['mae']:>12.2f}")
    print(f"{'MAPE':<10} {cnn_metrics['mape']:>11.2f}% {ext_metrics['mape']:>11.2f}%")
    print("=" * 60)

    # --- Create figure ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # PLOT 1: Representative 48h forecast (3 curves)
    ax1 = axes[0, 0]
    sample_idx = len(y_test) // 2
    actual = y_test[sample_idx]
    predicted = y_pred[sample_idx]
    external = fc_test[sample_idx]
    x = np.arange(FORECAST_HORIZON)

    ax1.plot(x, actual, 'b-', label='Actual (Load_Is)', linewidth=1.2)
    ax1.plot(x, predicted, 'r--', label='CNN-LSTM Prediction', linewidth=1.2)
    ext_plot_valid = np.isfinite(external) & (external > 0)
    if ext_plot_valid.any():
        ax1.plot(x[ext_plot_valid], external[ext_plot_valid], 'g-.',
                 label='External Forecast', linewidth=1.2)
    ax1.set_title(f'48h Forecast from {pd.Timestamp(ts_test[sample_idx]).strftime("%Y-%m-%d %H:%M")}',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time step (5-min intervals)')
    ax1.set_ylabel('Load (kW)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Add hour ticks
    tick_pos = np.arange(0, FORECAST_HORIZON, 72)  # every 6 hours
    tick_labels = [f'+{int(p * 5 / 60)}h' for p in tick_pos]
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labels)

    # PLOT 2: KPI bar chart comparison
    ax2 = axes[0, 1]
    metric_names = ['RMSE (kW)', 'MAE (kW)', 'MAPE (%)']
    cnn_vals = [cnn_metrics['rmse'], cnn_metrics['mae'], cnn_metrics['mape']]
    ext_vals = [ext_metrics['rmse'], ext_metrics['mae'], ext_metrics['mape']]
    x_pos = np.arange(len(metric_names))
    width = 0.35

    bars_cnn = ax2.bar(x_pos - width / 2, cnn_vals, width,
                       label='CNN-LSTM', color='steelblue', edgecolor='black', alpha=0.8)
    bars_ext = ax2.bar(x_pos + width / 2, ext_vals, width,
                       label='External Forecast', color='coral', edgecolor='black', alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metric_names, fontsize=11)
    ax2.set_title('Model Comparison: KPIs', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar_group in [bars_cnn, bars_ext]:
        for bar in bar_group:
            h = bar.get_height()
            if np.isfinite(h):
                ax2.text(bar.get_x() + bar.get_width() / 2, h,
                         f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # PLOT 3: Training history
    ax3 = axes[1, 0]
    ax3.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax3.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax3.set_title('Training History', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MSE)')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # PLOT 4: Actual vs Predicted scatter
    ax4 = axes[1, 1]
    ax4.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.2, s=3, c='steelblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Prediction')
    ax4.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Actual Load (kW)')
    ax4.set_ylabel('Predicted Load (kW)')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'CNN_LSTM_Results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nResults plot saved to: {save_path}")


# =============================================================================
# SAVE / LOAD CONFIG
# =============================================================================

def save_config(scaler_X, scaler_y, scaler_stat, scaler_pv):
    """Save scalers and model configuration for later prediction."""
    np.savez(CONFIG_PATH,
             lookback_steps=LOOKBACK_STEPS,
             forecast_horizon=FORECAST_HORIZON,
             stat_n_weeks=STAT_N_WEEKS,
             feature_columns=np.array(INPUT_FEATURES),
             scaler_X_min_=scaler_X.min_,
             scaler_X_scale_=scaler_X.scale_,
             scaler_X_data_min_=scaler_X.data_min_,
             scaler_X_data_max_=scaler_X.data_max_,
             scaler_X_data_range_=scaler_X.data_range_,
             scaler_y_min_=scaler_y.min_,
             scaler_y_scale_=scaler_y.scale_,
             scaler_y_data_min_=scaler_y.data_min_,
             scaler_y_data_max_=scaler_y.data_max_,
             scaler_y_data_range_=scaler_y.data_range_,
             scaler_stat_min_=scaler_stat.min_,
             scaler_stat_scale_=scaler_stat.scale_,
             scaler_stat_data_min_=scaler_stat.data_min_,
             scaler_stat_data_max_=scaler_stat.data_max_,
             scaler_stat_data_range_=scaler_stat.data_range_,
             scaler_pv_min_=scaler_pv.min_,
             scaler_pv_scale_=scaler_pv.scale_,
             scaler_pv_data_min_=scaler_pv.data_min_,
             scaler_pv_data_max_=scaler_pv.data_max_,
             scaler_pv_data_range_=scaler_pv.data_range_)
    print(f"Config saved to: {CONFIG_PATH}")


def load_config():
    """Load scalers and model configuration."""
    config = np.load(CONFIG_PATH, allow_pickle=True)

    scaler_X = MinMaxScaler()
    scaler_X.min_ = config['scaler_X_min_']
    scaler_X.scale_ = config['scaler_X_scale_']
    scaler_X.data_min_ = config['scaler_X_data_min_']
    scaler_X.data_max_ = config['scaler_X_data_max_']
    scaler_X.data_range_ = config['scaler_X_data_range_']
    scaler_X.n_features_in_ = len(config['scaler_X_min_'])

    scaler_y = MinMaxScaler()
    scaler_y.min_ = config['scaler_y_min_']
    scaler_y.scale_ = config['scaler_y_scale_']
    scaler_y.data_min_ = config['scaler_y_data_min_']
    scaler_y.data_max_ = config['scaler_y_data_max_']
    scaler_y.data_range_ = config['scaler_y_data_range_']
    scaler_y.n_features_in_ = len(config['scaler_y_min_'])

    scaler_stat = MinMaxScaler()
    scaler_stat.min_ = config['scaler_stat_min_']
    scaler_stat.scale_ = config['scaler_stat_scale_']
    scaler_stat.data_min_ = config['scaler_stat_data_min_']
    scaler_stat.data_max_ = config['scaler_stat_data_max_']
    scaler_stat.data_range_ = config['scaler_stat_data_range_']
    scaler_stat.n_features_in_ = len(config['scaler_stat_min_'])

    scaler_pv = MinMaxScaler()
    scaler_pv.min_ = config['scaler_pv_min_']
    scaler_pv.scale_ = config['scaler_pv_scale_']
    scaler_pv.data_min_ = config['scaler_pv_data_min_']
    scaler_pv.data_max_ = config['scaler_pv_data_max_']
    scaler_pv.data_range_ = config['scaler_pv_data_range_']
    scaler_pv.n_features_in_ = len(config['scaler_pv_min_'])

    return scaler_X, scaler_y, scaler_stat, scaler_pv


# =============================================================================
# TRAIN MODE
# =============================================================================

def run_train():
    """Train the CNN-LSTM model, evaluate on held-out test set, save model."""
    # Fix random seed for reproducible training (avoids local-minimum lottery)
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("=" * 70)
    print("CNN-LSTM 48h LOAD PREDICTION - TRAINING")
    print("=" * 70)

    # 1. Load PV correction table + data
    print("\n--- Step 1: Loading Data ---")
    ratio_table = load_pv_correction_table()
    df = load_data(ratio_table=ratio_table)

    # 2. Create sequences (only from rows with valid Load_Is)
    print("\n--- Step 2: Creating Sequences ---")
    print(f"  Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.0f}h)")
    print(f"  Forecast: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * 5 / 60:.0f}h)")
    print(f"  Step size: {STEP_SIZE} ({STEP_SIZE * 5}min)")

    X, y, timestamps, forecast_loads, start_indices, pv_horizons = create_sequences(
        df, step=STEP_SIZE)

    if len(X) == 0:
        print("ERROR: No valid sequences could be created. Check data.")
        return

    # 2b. Build statistical profiles for decoder conditioning (Input 2)
    print("\n--- Step 2b: Building Statistical Profiles ---")
    load_values = df['Load_Is'].values.astype(np.float32)
    stat_profiles = build_stat_profiles(load_values, start_indices)
    print(f"  Stat profiles shape: {stat_profiles.shape}  "
          f"(samples, horizon, {STAT_N_WEEKS} weeks)")

    # 3. Train/test split by time
    print("\n--- Step 3: Train/Test Split ---")
    last_date = pd.Timestamp(timestamps[-1])
    test_cutoff = last_date - pd.Timedelta(days=TEST_DAYS)

    mask_train = timestamps < np.datetime64(test_cutoff)
    mask_test = timestamps >= np.datetime64(test_cutoff)

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    stat_train = stat_profiles[mask_train]
    stat_test = stat_profiles[mask_test]
    pv_train = pv_horizons[mask_train]
    pv_test  = pv_horizons[mask_test]
    ts_test = timestamps[mask_test]
    fc_test = forecast_loads[mask_test]

    print(f"  Test cutoff: {test_cutoff.strftime('%Y-%m-%d')}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Not enough data for train/test split.")
        return

    # 4. Scale features, targets, stat profiles, and PV horizons
    print("\n--- Step 4: Scaling ---")
    n_train, n_steps, n_feat = X_train.shape
    n_test = X_test.shape[0]

    # Fit scaler on raw feature columns, then apply via broadcasting.
    # Using reshape(-1, n_feat) on the windowed array would allocate ~10 GB for
    # long datasets (n_train * n_steps rows); broadcasting over the 3-D array
    # is mathematically identical and uses only the working array memory.
    scaler_X = MinMaxScaler()
    scaler_X.fit(df[INPUT_FEATURES].dropna().values.astype(np.float32))
    # MinMaxScaler formula: X_scaled = X * scale_ + min_  (both shape (n_feat,))
    # Scale in-place to avoid creating a second large copy of the array.
    X_train *= scaler_X.scale_
    X_train += scaler_X.min_
    X_train_s = X_train          # already scaled, same array
    X_test *= scaler_X.scale_
    X_test += scaler_X.min_
    X_test_s = X_test            # already scaled, same array

    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train)

    # Scale stat profiles: reshape (N, 576, 4) → (N*576, 4), scale, reshape back
    scaler_stat = MinMaxScaler()
    stat_train_s = scaler_stat.fit_transform(
        stat_train.reshape(-1, STAT_N_WEEKS)).reshape(n_train, FORECAST_HORIZON, STAT_N_WEEKS)
    stat_test_s = scaler_stat.transform(
        stat_test.reshape(-1, STAT_N_WEEKS)).reshape(n_test, FORECAST_HORIZON, STAT_N_WEEKS)

    # Scale PV horizons: reshape (N, 576) → (N*576, 1), scale, reshape to (N, 576, 1)
    scaler_pv = MinMaxScaler()
    pv_train_s = scaler_pv.fit_transform(
        pv_train.reshape(-1, 1)).reshape(n_train, FORECAST_HORIZON, 1)
    pv_test_s = scaler_pv.transform(
        pv_test.reshape(-1, 1)).reshape(n_test, FORECAST_HORIZON, 1)

    print(f"  Feature range: [{scaler_X.data_min_.min():.1f}, {scaler_X.data_max_.max():.1f}]")
    print(f"  Target range:  [{scaler_y.data_min_.min():.1f}, {scaler_y.data_max_.max():.1f}]")
    print(f"  Stat range:    [{scaler_stat.data_min_.min():.1f}, {scaler_stat.data_max_.max():.1f}]")
    print(f"  PV range:      [{scaler_pv.data_min_.min():.1f}, {scaler_pv.data_max_.max():.1f}] kW")

    # 5. Build model (triple input: encoder + stat + PV conditioning)
    print("\n--- Step 5: Building Model ---")
    model = build_model(
        input_shape=(n_steps, n_feat),
        stat_shape=(FORECAST_HORIZON, STAT_N_WEEKS),
        pv_shape=(FORECAST_HORIZON, 1))
    model.summary()

    # 6. Train with triple inputs
    print("\n--- Step 6: Training ---")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
    ]

    start_time = time.time()
    history = model.fit(
        [X_train_s, stat_train_s, pv_train_s], y_train_s,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.0f}s ({len(history.history['loss'])} epochs)")

    # 7. Predict on test set
    print("\n--- Step 7: Evaluating on Test Set ---")
    y_pred_s = model.predict([X_test_s, stat_test_s, pv_test_s], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s)

    # 8. Visualize and compare
    plot_evaluation(y_test, y_pred, fc_test, ts_test, history)

    # 9. Save config (including all scalers)
    print("\n--- Step 8: Saving Model & Config ---")
    save_config(scaler_X, scaler_y, scaler_stat, scaler_pv)
    print(f"  Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


# =============================================================================
# PREDICT MODE
# =============================================================================

def run_predict(date_str, n_weeks=HYBRID_N_PAST_WEEKS, agg_method=HYBRID_AGG_METHOD):
    """
    Load saved model and predict 48h from a given date.

    Produces three forecasts:
      1. CNN-LSTM (AI model)
      2. Statistical baseline (mean of last N same-weekday values)
      3. External forecast (from data file)

    Parameters
    ----------
    date_str : str
        Prediction start date in format YYYY-MM-DD or YYYY-MM-DD HH:MM
    n_weeks : int
        Number of same-day-of-week weeks to look back for statistical baseline.
    agg_method : str
        'mean' or 'median' for aggregating historical same-weekday values.
    """
    print("=" * 70)
    print(f"CNN-LSTM 48h LOAD PREDICTION - PREDICT FROM {date_str}")
    print("=" * 70)

    # 1. Check model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        print("ERROR: Model not found. Run --train first.")
        print(f"  Expected: {MODEL_PATH}")
        print(f"  Expected: {CONFIG_PATH}")
        return

    # 2. Load model and config
    print("\n--- Loading model ---")
    model = load_model(MODEL_PATH)
    scaler_X, scaler_y, scaler_stat, scaler_pv = load_config()
    n_feat = len(INPUT_FEATURES)

    # 3. Load PV correction table + data
    print("\n--- Loading data ---")
    ratio_table = load_pv_correction_table()
    df = load_data(ratio_table=ratio_table)

    # 4. Find prediction start point
    pred_date = pd.to_datetime(date_str)
    idx = (df['DateTime'] - pred_date).abs().idxmin()
    actual_start = df['DateTime'].iloc[idx]
    print(f"  Prediction starts at: {actual_start}")

    if idx < LOOKBACK_STEPS:
        print(f"ERROR: Not enough history. Need {LOOKBACK_STEPS} steps before {date_str}.")
        print(f"  Available: {idx} steps. Earliest possible date: "
              f"{df['DateTime'].iloc[LOOKBACK_STEPS]}")
        return

    # 5. Extract encoder input (7 days before prediction point)
    input_df = df.iloc[idx - LOOKBACK_STEPS:idx]
    X = input_df[INPUT_FEATURES].values.astype(np.float32)

    if np.isnan(X).any():
        nan_cols = [INPUT_FEATURES[c] for c in range(X.shape[1]) if np.isnan(X[:, c]).any()]
        print(f"WARNING: NaN in input features: {nan_cols}. Filling with interpolation.")
        X = pd.DataFrame(X, columns=INPUT_FEATURES).interpolate().bfill().ffill().values

    # 5b. Build statistical profile for decoder conditioning (Input 2)
    load_values = df['Load_Is'].values.astype(np.float32)
    stat_profile = build_stat_profiles(load_values, np.array([idx]),
                                        horizon=FORECAST_HORIZON,
                                        n_weeks=STAT_N_WEEKS)  # (1, 576, 4)
    print(f"  Stat profile built: {stat_profile.shape}")

    # 5c. Extract PV estimate for forecast horizon (Input 3)
    # PV_Est was already computed in load_data() via the correction table
    end_pv = min(idx + FORECAST_HORIZON, len(df))
    pv_slice = df['PV_Est'].values[idx:end_pv].astype(np.float32)
    # Pad with zeros if forecast period extends beyond data
    if len(pv_slice) < FORECAST_HORIZON:
        pv_slice = np.pad(pv_slice, (0, FORECAST_HORIZON - len(pv_slice)))
    pv_profile = pv_slice.reshape(1, FORECAST_HORIZON)   # (1, 576)
    print(f"  PV estimate built: max={pv_slice.max():.1f} kW, "
          f"mean_daytime={pv_slice[pv_slice > 0].mean():.1f} kW"
          if pv_slice.max() > 0 else "  PV estimate: 0 kW (nighttime / no irradiance data)")

    # 6. Scale and predict (triple inputs)
    X_scaled = scaler_X.transform(X.reshape(-1, n_feat)).reshape(1, LOOKBACK_STEPS, n_feat)
    stat_scaled = scaler_stat.transform(
        stat_profile.reshape(-1, STAT_N_WEEKS)).reshape(1, FORECAST_HORIZON, STAT_N_WEEKS)
    pv_scaled = scaler_pv.transform(
        pv_profile.reshape(-1, 1)).reshape(1, FORECAST_HORIZON, 1)
    y_pred_s = model.predict([X_scaled, stat_scaled, pv_scaled], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s).flatten()

    # 7. Generate timestamps
    start_time = df['DateTime'].iloc[idx]
    future_ts = [start_time + pd.Timedelta(minutes=5 * i)
                 for i in range(FORECAST_HORIZON)]

    # 7b. Compute statistical baseline (4-week same-weekday mean)
    print(f"\n--- Computing Statistical Baseline ---")
    print(f"  Aggregation: {agg_method} | Past weeks: {n_weeks}")
    stat_values = compute_stat_baseline(
        df, future_ts, n_past_weeks=n_weeks, agg_method=agg_method)

    # 8. Get actual and external forecast if available in the dataset
    end_idx = min(idx + FORECAST_HORIZON, len(df))
    future_df = df.iloc[idx:end_idx]
    actual_load = pd.to_numeric(future_df['Load_Is'], errors='coerce').values
    external_load = pd.to_numeric(future_df['Forecast_Load'], errors='coerce').values

    # 9. Compute KPIs and Plot
    has_actual = len(actual_load) > 0 and np.isfinite(actual_load).any()
    ext_valid = np.isfinite(external_load) & (external_load > 0)

    # Compute metrics on LAST 24h only (fair comparison: external forecasts 1 day ahead)
    KPI_START = 288  # step 288 = +24h mark
    cnn_m = None
    stat_m = None
    ext_m = None
    if has_actual and len(actual_load) > KPI_START:
        kpi_actual = actual_load[KPI_START:]
        kpi_len = len(kpi_actual)
        cnn_m = compute_metrics(kpi_actual, y_pred[KPI_START:KPI_START + kpi_len])
        stat_m = compute_metrics(kpi_actual, stat_values[KPI_START:KPI_START + kpi_len])
        kpi_ext = external_load[KPI_START:] if len(external_load) > KPI_START else np.array([])
        kpi_ext_valid = np.isfinite(kpi_ext) & (kpi_ext > 0)
        if kpi_ext_valid.any():
            ext_m = compute_metrics(kpi_actual[kpi_ext_valid[:kpi_len]],
                                     kpi_ext[kpi_ext_valid[:len(kpi_ext)]])

    # --- PLOT: Forecast + KPIs + Weight Profile ---
    tick_pos = np.arange(0, FORECAST_HORIZON, 72)  # every 6 hours
    tick_labels = [(start_time + pd.Timedelta(minutes=5 * p)).strftime('%m-%d %H:%M')
                   for p in tick_pos]

    if has_actual and cnn_m is not None:
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])   # Forecast curves
        ax2 = fig.add_subplot(gs[0, 1])   # KPI bars
    else:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = None

    # --- Left: Forecast curves ---
    x = np.arange(FORECAST_HORIZON)

    if has_actual:
        ax1.plot(x[:len(actual_load)], actual_load, 'b-',
                 label='Actual (Load_Is)', linewidth=1.5)

    ax1.plot(x, y_pred, 'r-', label='CNN-LSTM (AI)', linewidth=1.5, alpha=0.7)
    ax1.plot(x, stat_values, 'c--',
             label=f'Statistical ({agg_method}, {n_weeks}w)',
             linewidth=1.0, alpha=0.5)

    # Overlay PV estimate (on secondary y-axis to avoid scale clash)
    if pv_slice.max() > 0:
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(x, 0, pv_slice, alpha=0.15, color='gold')
        ax1_twin.plot(x, pv_slice, color='gold', linewidth=0.8, alpha=0.6,
                      label='PV Est (kW)')
        ax1_twin.set_ylabel('PV Est (kW)', fontsize=9, color='goldenrod')
        ax1_twin.tick_params(axis='y', labelcolor='goldenrod', labelsize=8)
        ax1_twin.legend(fontsize=8, loc='upper left')

    if ext_valid.any():
        ax1.plot(x[:len(external_load)][ext_valid[:len(external_load)]],
                 external_load[ext_valid[:len(external_load)]],
                 'g-.', label='External Forecast', linewidth=1.5)

    ax1.set_title(f'48h Load Prediction from {date_str}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Load (kW)', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

    # --- Right: KPI bar chart (CNN-LSTM vs Statistical vs External) ---
    if ax2 is not None and cnn_m is not None:
        metric_names = ['RMSE (kW)', 'MAE (kW)', 'MAPE (%)']
        cnn_vals = [cnn_m['rmse'], cnn_m['mae'], cnn_m['mape']]
        stat_vals = [stat_m['rmse'], stat_m['mae'], stat_m['mape']]
        x_pos = np.arange(len(metric_names))

        if ext_m is not None:
            ext_vals = [ext_m['rmse'], ext_m['mae'], ext_m['mape']]
            width = 0.25
            bars_cnn = ax2.bar(x_pos - width, cnn_vals, width,
                               label='CNN-LSTM', color='steelblue', edgecolor='black', alpha=0.8)
            bars_stat = ax2.bar(x_pos, stat_vals, width,
                                label=f'Stats ({agg_method})', color='cyan', edgecolor='black', alpha=0.8)
            bars_ext = ax2.bar(x_pos + width, ext_vals, width,
                               label='External', color='coral', edgecolor='black', alpha=0.8)
            all_bars = [bars_cnn, bars_stat, bars_ext]
        else:
            width = 0.3
            bars_cnn = ax2.bar(x_pos - width / 2, cnn_vals, width,
                               label='CNN-LSTM', color='steelblue', edgecolor='black', alpha=0.8)
            bars_stat = ax2.bar(x_pos + width / 2, stat_vals, width,
                                label=f'Stats ({agg_method})', color='cyan', edgecolor='black', alpha=0.8)
            all_bars = [bars_cnn, bars_stat]

        for bar_group in all_bars:
            for bar in bar_group:
                h = bar.get_height()
                if np.isfinite(h):
                    ax2.text(bar.get_x() + bar.get_width() / 2, h,
                             f'{h:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metric_names, fontsize=11)
        ax2.set_title('Prediction KPIs (last 24h)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'Prediction_{date_str.replace(":", "-")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to: {plot_path}")

    # 10. Print KPIs to console as well
    if has_actual and cnn_m is not None:
        print(f"\nCNN-LSTM  | RMSE: {cnn_m['rmse']:.2f} kW, "
              f"MAE: {cnn_m['mae']:.2f} kW, MAPE: {cnn_m['mape']:.2f}%")
        if stat_m is not None:
            print(f"Stats     | RMSE: {stat_m['rmse']:.2f} kW, "
                  f"MAE: {stat_m['mae']:.2f} kW, MAPE: {stat_m['mape']:.2f}%")
        if ext_m is not None:
            print(f"External  | RMSE: {ext_m['rmse']:.2f} kW, "
                  f"MAE: {ext_m['mae']:.2f} kW, MAPE: {ext_m['mape']:.2f}%")
    else:
        print("\nNo actual load data available for this period (future prediction).")

    # 11. Save predictions to CSV
    pred_df = pd.DataFrame({
        'DateTime': future_ts,
        'CNN_LSTM_Load_kW': y_pred,
        'Statistical_Load_kW': stat_values,
        'PV_Est_kW': pv_slice,
    })
    csv_path = os.path.join(RESULTS_DIR, f'Prediction_{date_str.replace(":", "-")}.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")

    # Summary
    print(f"\n--- Prediction Summary ---")
    print(f"  Period: {future_ts[0]} to {future_ts[-1]}")
    print(f"  CNN-LSTM  -> Mean: {y_pred.mean():.1f} kW, "
          f"Min: {y_pred.min():.1f}, Max: {y_pred.max():.1f}")
    print(f"  Stats     -> Mean: {np.nanmean(stat_values):.1f} kW")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CNN-LSTM 48h Load Prediction',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  python CNN_LSTM_Prediction.py --train
  python CNN_LSTM_Prediction.py --predict 2026-02-15
  python CNN_LSTM_Prediction.py --predict "2026-02-15 08:00"
  python CNN_LSTM_Prediction.py --predict 2026-02-15 --n-weeks 6 --agg-method median
""")
    parser.add_argument('--train', action='store_true',
                        help='Train the model on historical data')
    parser.add_argument('--predict', type=str, metavar='DATE',
                        help='Predict 48h from DATE (format: YYYY-MM-DD)')
    parser.add_argument('--n-weeks', type=int, default=HYBRID_N_PAST_WEEKS,
                        help=f'Same-weekday weeks to look back (default: {HYBRID_N_PAST_WEEKS})')
    parser.add_argument('--agg-method', type=str, default=HYBRID_AGG_METHOD,
                        choices=['mean', 'median'],
                        help=f'Statistical aggregation method (default: {HYBRID_AGG_METHOD})')

    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.predict:
        run_predict(args.predict,
                    n_weeks=args.n_weeks,
                    agg_method=args.agg_method)
    else:
        parser.print_help()
