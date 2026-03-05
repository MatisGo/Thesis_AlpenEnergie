"""
CNN-LSTM 48h Load Prediction
=============================
Hybrid CNN-LSTM model for 48-hour ahead electricity load forecasting.

Architecture (based on Chung & Jang 2022, Khan et al. 2020):
  Encoder:
    - 3x Conv1D blocks with aggressive pooling (2016 -> 63 timesteps)
    - 2x LSTM layers for temporal pattern learning
  Decoder conditioning:
    - Same-weekday statistical profiles (past 4 weeks) for the forecast horizon
    - Reshaped to hourly blocks, processed with Dense, concatenated with encoder output
  Decoder:
    - RepeatVector + LSTM decoder + TimeDistributed Dense
    - Outputs (48 hours, 12 steps each) then flattened to 576 steps

Input 1 - Encoder (7-day lookback, 2016 steps x 10 features):
  Load_Is (past), Load_yesterday, Load_last_week,
  Hour_sin, Hour_cos, Weekday_sin, Weekday_cos,
  PHolyday, Temp_Forecast, Rain_Forecast

Input 2 - Decoder conditioning (576 steps x 4 weekly profiles):
  Load from same weekday 1w, 2w, 3w, 4w ago for each forecast step

Output: Load_Is for the next 48 hours (576 steps at 5-min resolution)
Loss:   Huber (less sensitive to spikes than MSE)

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'Data_Prediction.xlsx')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'CNN_LSTM_Model.keras')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'CNN_LSTM_Config.npz')

# Data parameters
LOOKBACK_STEPS = 2016       # 7 days * 288 steps/day (5-min resolution)
FORECAST_HORIZON = 576      # 2 days * 288 steps/day
STEP_SIZE = 1               # Sample every 30 min (reduces memory, keeps enough samples)
METADATA_ROWS = 2           # Rows to skip in xlsx (Einheit, Signalname)
TEST_DAYS = 10              # Last N days held out for testing

# Input features (order matters - must match during prediction)
INPUT_FEATURES = [
    'Load_Is',              # Past load values
    'Load_yesterday',       # Load at same time 24h ago (lagged feature)
    'Load_last_week',       # Load at same time 7 days ago (lagged feature)
    'Hour_sin',             # Time of day (sin component)
    'Hour_cos',             # Time of day (cos component)
    'Weekday_sin',          # Day of week (sin component)
    'Weekday_cos',          # Day of week (cos component)
    'PHolyday',             # Public holiday flag (0/1)
    'Temp_Forecast',        # Temperature forecast (deg C)
    'Rain_Forecast',        # Rain forecast (mm)
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

# Hybrid AI + Statistical blending
HYBRID_N_PAST_WEEKS = 4        # Number of same-day-of-week weeks to look back
HYBRID_CV_THRESHOLD = 0.5      # CV at which AI gets 100% weight (adjustable)
HYBRID_AGG_METHOD = 'mean'     # 'mean' or 'median' for historical aggregation

# Decoder conditioning (same-weekday statistical profiles as second model input)
STAT_N_WEEKS = 4               # Number of past same-weekday profiles fed to decoder


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data():
    """
    Load and preprocess Data_Prediction.xlsx.

    Skips metadata rows, parses datetime, creates cyclical time encodings,
    and handles missing values via interpolation.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed data with all required features.
    """
    print("Loading data from:", DATA_PATH)

    df = pd.read_excel(DATA_PATH, header=0)
    df = df.iloc[METADATA_ROWS:].reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]

    # Parse datetime
    df['DateTime'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S',
                                     errors='coerce')

    # Convert numeric columns
    numeric_cols = ['Load_Is', 'Forecast_Load', 'Temp_Forecast', 'Rain_Forecast',
                    'Weekday', 'PHolyday']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse Day_Time → fractional hour → cyclical encoding
    df['Day_Time_td'] = pd.to_timedelta(df['Day_Time'].astype(str), errors='coerce')
    df['HourFrac'] = df['Day_Time_td'].dt.total_seconds() / 3600.0
    df['Hour_sin'] = np.sin(2 * np.pi * df['HourFrac'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['HourFrac'] / 24)

    # Weekday cyclical encoding
    df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

    # Drop rows without valid datetime, sort chronologically
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    # Interpolate missing values in continuous features
    for col in ['Load_Is', 'Temp_Forecast', 'Rain_Forecast']:
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # Fill PHolyday NaN with 0 (assume no holiday if missing)
    df['PHolyday'] = df['PHolyday'].fillna(0)

    # --- Lagged features ---
    # Load at the same time yesterday (288 steps = 24h at 5-min)
    df['Load_yesterday'] = df['Load_Is'].shift(288)
    # Load at the same time last week (2016 steps = 7 days at 5-min)
    df['Load_last_week'] = df['Load_Is'].shift(2016)
    # Fill initial NaN from shifts with backward fill
    df['Load_yesterday'] = df['Load_yesterday'].bfill()
    df['Load_last_week'] = df['Load_last_week'].bfill()

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"  Rows with Load_Is: {df['Load_Is'].notna().sum()}")

    return df


# =============================================================================
# SEQUENCE CREATION
# =============================================================================

def create_sequences(df, step=STEP_SIZE):
    """
    Create sliding-window input/output sequences.

    Each sample:
      X = 7 days of features BEFORE the prediction point
      y = 48 hours of Load_Is AFTER the prediction point

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data (must have INPUT_FEATURES, TARGET, DateTime, Forecast_Load).
    step : int
        Sliding window step size (in 5-min increments).

    Returns
    -------
    X, y, timestamps, forecast_loads : np.ndarray
    """
    features = df[INPUT_FEATURES].values
    targets = df[TARGET].values
    timestamps = df['DateTime'].values
    forecast_loads = pd.to_numeric(df['Forecast_Load'], errors='coerce').values

    X, y, ts, fc, idx_list = [], [], [], [], []

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

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    ts = np.array(ts)
    fc = np.array(fc, dtype=np.float32)
    start_indices = np.array(idx_list, dtype=np.int64)

    print(f"  Created {len(X)} sequences")
    print(f"  X shape: {X.shape}  (samples, lookback, features)")
    print(f"  y shape: {y.shape}  (samples, forecast_horizon)")

    return X, y, ts, fc, start_indices


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

def build_model(input_shape, stat_shape, output_hours=OUTPUT_HOURS,
                steps_per_hour=STEPS_PER_HOUR):
    """
    Build CNN-LSTM model with decoder conditioning.

    Dual-input architecture:
      Input 1 (encoder): 7-day lookback features → CNN → LSTM → encoded context
      Input 2 (decoder cond.): Same-weekday load profiles for forecast horizon

    The decoder receives both the encoded context (what the model learned)
    and the statistical profiles (what typically happens on this weekday),
    allowing it to anchor predictions to recurring weekly patterns.

    Parameters
    ----------
    input_shape : tuple
        Encoder input: (timesteps, features) = (2016, 10)
    stat_shape : tuple
        Statistical profile: (forecast_horizon, n_weeks) = (576, 4)
    output_hours : int
        Number of hourly blocks = 48
    steps_per_hour : int
        5-min steps per hour = 12

    Returns
    -------
    model : keras.Model
    """
    # --- Two inputs ---
    encoder_input = Input(shape=input_shape, name='encoder_input')
    stat_input = Input(shape=stat_shape, name='stat_input')

    # --- CNN Encoder (3 blocks: 2016 -> 504 -> 126 -> 63) ---
    x = encoder_input
    for i, (filters, kernel, pool) in enumerate(zip(CNN_FILTERS, CNN_KERNELS, POOL_SIZES)):
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

    # --- Process statistical profiles ---
    # Reshape (576, n_weeks) → (48, 12 * n_weeks) to match hourly decoder blocks
    n_weeks = stat_shape[1]
    stat = Reshape((output_hours, steps_per_hour * n_weeks))(stat_input)  # (batch, 48, 48)
    stat = TimeDistributed(Dense(32, activation='relu'))(stat)            # (batch, 48, 32)

    # --- Concatenate encoder context + statistical conditioning ---
    x = Concatenate()([x, stat])  # (batch, 48, 256 + 32 = 288)

    # LSTM decoder processes each hour with both context and weekly pattern
    x = LSTM(128, return_sequences=True)(x)  # (batch, 48, 128)
    x = Dropout(DROPOUT_RATE)(x)

    # TimeDistributed Dense: each hour -> 12 five-min steps
    x = TimeDistributed(Dense(64, activation='relu'))(x)  # (batch, 48, 64)
    x = TimeDistributed(Dense(steps_per_hour))(x)          # (batch, 48, 12)

    # Flatten to (batch, 576)
    outputs = Reshape((output_hours * steps_per_hour,))(x)

    model = Model(inputs=[encoder_input, stat_input], outputs=outputs)

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
# HYBRID AI + STATISTICAL PREDICTION
# =============================================================================

def compute_hybrid_prediction(df, ai_prediction, forecast_timestamps,
                               n_past_weeks=HYBRID_N_PAST_WEEKS,
                               cv_threshold=HYBRID_CV_THRESHOLD,
                               agg_method=HYBRID_AGG_METHOD):
    """
    Blend AI prediction with statistical same-day-of-week baseline.

    For each 5-min slot in the forecast:
      1. Find the same weekday+time for the last n_past_weeks weeks
      2. Compute statistical value (mean or median) and CV (std/mean)
      3. Blend: low CV (consistent pattern) → trust statistics
               high CV (variable)          → trust AI

    Weight formula:
      weight_AI  = min(1.0, CV / cv_threshold)
      weight_stat = 1.0 - weight_AI
      hybrid = weight_stat * stat_value + weight_AI * ai_value

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DateTime and Load_Is columns.
    ai_prediction : np.ndarray
        AI model prediction, shape (FORECAST_HORIZON,).
    forecast_timestamps : list of pd.Timestamp
        Timestamps for each forecast step.
    n_past_weeks : int
        Number of same-day-of-week weeks to look back.
    cv_threshold : float
        CV value at which AI gets 100% weight.
    agg_method : str
        'mean' or 'median' for aggregating historical values.

    Returns
    -------
    hybrid_pred : np.ndarray  – blended prediction
    stat_values : np.ndarray  – pure statistical baseline
    cv_values   : np.ndarray  – coefficient of variation per slot
    weights_ai  : np.ndarray  – AI weight per slot (0 = all stats, 1 = all AI)
    """
    n_steps = len(ai_prediction)
    stat_values = np.full(n_steps, np.nan)
    cv_values = np.full(n_steps, np.nan)

    # Build DatetimeIndex-based Series for fast nearest-neighbour lookup
    df_load = df.set_index('DateTime')['Load_Is'].copy()
    df_load = df_load[~df_load.index.duplicated(keep='first')].sort_index()

    for i, ts in enumerate(forecast_timestamps):
        ts = pd.Timestamp(ts)
        historical_values = []

        for w in range(1, n_past_weeks + 1):
            hist_ts = ts - pd.Timedelta(weeks=w)
            # Find closest timestamp in data (tolerance: 5 min)
            idx_arr = df_load.index.get_indexer([hist_ts], method='nearest',
                                                 tolerance=pd.Timedelta(minutes=5))
            if idx_arr[0] >= 0:
                val = df_load.iloc[idx_arr[0]]
                if np.isfinite(val) and val > 0:
                    historical_values.append(val)

        if len(historical_values) >= 2:
            hist_arr = np.array(historical_values)
            # Aggregate: mean or median (mark in code – test both)
            if agg_method == 'median':
                stat_values[i] = np.median(hist_arr)   # robust to outliers
            else:
                stat_values[i] = np.mean(hist_arr)     # uses all information

            hist_mean = np.mean(hist_arr)
            hist_std = np.std(hist_arr, ddof=0)
            cv_values[i] = hist_std / hist_mean if hist_mean > 0 else 1.0
        elif len(historical_values) == 1:
            stat_values[i] = historical_values[0]
            cv_values[i] = 0.0   # single value → assume consistent
        else:
            # No historical data available → fall back to AI
            stat_values[i] = ai_prediction[i]
            cv_values[i] = 1.0

    # Compute weights (smooth transition)
    weights_ai = np.minimum(1.0, cv_values / cv_threshold)
    weights_stat = 1.0 - weights_ai

    # Blend
    hybrid_pred = weights_stat * stat_values + weights_ai * ai_prediction

    return hybrid_pred, stat_values, cv_values, weights_ai


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
    save_path = os.path.join(SCRIPT_DIR, 'CNN_LSTM_Results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nResults plot saved to: {save_path}")


# =============================================================================
# SAVE / LOAD CONFIG
# =============================================================================

def save_config(scaler_X, scaler_y, scaler_stat):
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
             scaler_stat_data_range_=scaler_stat.data_range_)
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

    return scaler_X, scaler_y, scaler_stat


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

    # 1. Load data
    print("\n--- Step 1: Loading Data ---")
    df = load_data()

    # 2. Create sequences (only from rows with valid Load_Is)
    print("\n--- Step 2: Creating Sequences ---")
    print(f"  Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.0f}h)")
    print(f"  Forecast: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * 5 / 60:.0f}h)")
    print(f"  Step size: {STEP_SIZE} ({STEP_SIZE * 5}min)")

    X, y, timestamps, forecast_loads, start_indices = create_sequences(df, step=STEP_SIZE)

    if len(X) == 0:
        print("ERROR: No valid sequences could be created. Check data.")
        return

    # 2b. Build statistical profiles for decoder conditioning
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
    ts_test = timestamps[mask_test]
    fc_test = forecast_loads[mask_test]

    print(f"  Test cutoff: {test_cutoff.strftime('%Y-%m-%d')}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Not enough data for train/test split.")
        return

    # 4. Scale features, targets, and stat profiles
    print("\n--- Step 4: Scaling ---")
    n_train, n_steps, n_feat = X_train.shape
    n_test = X_test.shape[0]

    scaler_X = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(
        X_train.reshape(-1, n_feat)).reshape(n_train, n_steps, n_feat)
    X_test_s = scaler_X.transform(
        X_test.reshape(-1, n_feat)).reshape(n_test, n_steps, n_feat)

    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    # Scale stat profiles: reshape (N, 576, 4) → (N*576, 4), scale, reshape back
    scaler_stat = MinMaxScaler()
    stat_train_s = scaler_stat.fit_transform(
        stat_train.reshape(-1, STAT_N_WEEKS)).reshape(n_train, FORECAST_HORIZON, STAT_N_WEEKS)
    stat_test_s = scaler_stat.transform(
        stat_test.reshape(-1, STAT_N_WEEKS)).reshape(n_test, FORECAST_HORIZON, STAT_N_WEEKS)

    print(f"  Feature range: [{scaler_X.data_min_.min():.1f}, {scaler_X.data_max_.max():.1f}]")
    print(f"  Target range:  [{scaler_y.data_min_.min():.1f}, {scaler_y.data_max_.max():.1f}]")
    print(f"  Stat range:    [{scaler_stat.data_min_.min():.1f}, {scaler_stat.data_max_.max():.1f}]")

    # 5. Build model (dual input: encoder + stat conditioning)
    print("\n--- Step 5: Building Model ---")
    model = build_model(
        input_shape=(n_steps, n_feat),
        stat_shape=(FORECAST_HORIZON, STAT_N_WEEKS))
    model.summary()

    # 6. Train with dual inputs
    print("\n--- Step 6: Training ---")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
    ]

    start_time = time.time()
    history = model.fit(
        [X_train_s, stat_train_s], y_train_s,
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
    y_pred_s = model.predict([X_test_s, stat_test_s], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s)

    # 8. Visualize and compare
    plot_evaluation(y_test, y_pred, fc_test, ts_test, history)

    # 9. Save config (including stat scaler)
    print("\n--- Step 8: Saving Model & Config ---")
    save_config(scaler_X, scaler_y, scaler_stat)
    print(f"  Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


# =============================================================================
# PREDICT MODE
# =============================================================================

def run_predict(date_str, cv_threshold=HYBRID_CV_THRESHOLD,
                n_weeks=HYBRID_N_PAST_WEEKS, agg_method=HYBRID_AGG_METHOD):
    """
    Load saved model and predict 48h from a given date.

    Produces three forecasts:
      1. CNN-LSTM (pure AI)
      2. Statistical baseline (mean/median of last N same-weekday values)
      3. Hybrid blend (weighted by coefficient of variation)

    Parameters
    ----------
    date_str : str
        Prediction start date in format YYYY-MM-DD or YYYY-MM-DD HH:MM
    cv_threshold : float
        CV at which AI gets 100% weight (lower → more statistics).
    n_weeks : int
        Number of same-day-of-week weeks to look back.
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
    scaler_X, scaler_y, scaler_stat = load_config()
    n_feat = len(INPUT_FEATURES)

    # 3. Load data
    print("\n--- Loading data ---")
    df = load_data()

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

    # 5b. Build statistical profile for decoder conditioning
    load_values = df['Load_Is'].values.astype(np.float32)
    stat_profile = build_stat_profiles(load_values, np.array([idx]),
                                        horizon=FORECAST_HORIZON,
                                        n_weeks=STAT_N_WEEKS)  # (1, 576, 4)
    print(f"  Stat profile built: {stat_profile.shape}")

    # 6. Scale and predict (dual inputs)
    X_scaled = scaler_X.transform(X.reshape(-1, n_feat)).reshape(1, LOOKBACK_STEPS, n_feat)
    stat_scaled = scaler_stat.transform(
        stat_profile.reshape(-1, STAT_N_WEEKS)).reshape(1, FORECAST_HORIZON, STAT_N_WEEKS)
    y_pred_s = model.predict([X_scaled, stat_scaled], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s).flatten()

    # 7. Generate timestamps
    start_time = df['DateTime'].iloc[idx]
    future_ts = [start_time + pd.Timedelta(minutes=5 * i)
                 for i in range(FORECAST_HORIZON)]

    # 7b. Compute hybrid AI+Statistical prediction
    print(f"\n--- Computing Hybrid AI+Statistical Prediction ---")
    print(f"  Aggregation: {agg_method} | CV threshold: {cv_threshold} | Past weeks: {n_weeks}")
    hybrid_pred, stat_values, cv_values, weights_ai = compute_hybrid_prediction(
        df, y_pred, future_ts, n_past_weeks=n_weeks,
        cv_threshold=cv_threshold, agg_method=agg_method)
    print(f"  Mean AI weight: {weights_ai.mean():.2f}")
    print(f"  Steps with >50% statistics: {(weights_ai < 0.5).sum()}/{len(weights_ai)}")

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
    hybrid_m = None
    stat_m = None
    ext_m = None
    if has_actual and len(actual_load) > KPI_START:
        kpi_actual = actual_load[KPI_START:]
        kpi_len = len(kpi_actual)
        cnn_m = compute_metrics(kpi_actual, y_pred[KPI_START:KPI_START + kpi_len])
        hybrid_m = compute_metrics(kpi_actual, hybrid_pred[KPI_START:KPI_START + kpi_len])
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
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1],
                              hspace=0.35, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])   # Forecast curves
        ax2 = fig.add_subplot(gs[0, 1])   # KPI bars
        ax3 = fig.add_subplot(gs[1, :])   # Weight profile
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = None
        ax3 = fig.add_subplot(gs[1])

    # --- Top-left: Forecast curves ---
    x = np.arange(FORECAST_HORIZON)

    if has_actual:
        ax1.plot(x[:len(actual_load)], actual_load, 'b-',
                 label='Actual (Load_Is)', linewidth=1.5)

    ax1.plot(x, y_pred, 'r-', label='CNN-LSTM (AI)', linewidth=1.5, alpha=0.7)
    ax1.plot(x, hybrid_pred, 'm-', label='Hybrid AI+Stats', linewidth=2.0)
    ax1.plot(x, stat_values, 'c--',
             label=f'Statistical ({agg_method}, {n_weeks}w)',
             linewidth=1.0, alpha=0.5)

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

    # --- Top-right: KPI bar chart (CNN-LSTM vs Hybrid vs Statistical vs External) ---
    if ax2 is not None and cnn_m is not None:
        metric_names = ['RMSE (kW)', 'MAE (kW)', 'MAPE (%)']
        cnn_vals = [cnn_m['rmse'], cnn_m['mae'], cnn_m['mape']]
        hybrid_vals = [hybrid_m['rmse'], hybrid_m['mae'], hybrid_m['mape']]
        stat_vals = [stat_m['rmse'], stat_m['mae'], stat_m['mape']]
        x_pos = np.arange(len(metric_names))

        if ext_m is not None:
            ext_vals = [ext_m['rmse'], ext_m['mae'], ext_m['mape']]
            width = 0.18
            bars_cnn = ax2.bar(x_pos - 1.5 * width, cnn_vals, width,
                               label='CNN-LSTM', color='steelblue', edgecolor='black', alpha=0.8)
            bars_hyb = ax2.bar(x_pos - 0.5 * width, hybrid_vals, width,
                               label='Hybrid', color='orchid', edgecolor='black', alpha=0.8)
            bars_stat = ax2.bar(x_pos + 0.5 * width, stat_vals, width,
                                label=f'Stats ({agg_method})', color='cyan', edgecolor='black', alpha=0.8)
            bars_ext = ax2.bar(x_pos + 1.5 * width, ext_vals, width,
                               label='External', color='coral', edgecolor='black', alpha=0.8)
            all_bars = [bars_cnn, bars_hyb, bars_stat, bars_ext]
        else:
            width = 0.25
            bars_cnn = ax2.bar(x_pos - width, cnn_vals, width,
                               label='CNN-LSTM', color='steelblue', edgecolor='black', alpha=0.8)
            bars_hyb = ax2.bar(x_pos, hybrid_vals, width,
                               label='Hybrid', color='orchid', edgecolor='black', alpha=0.8)
            bars_stat = ax2.bar(x_pos + width, stat_vals, width,
                                label=f'Stats ({agg_method})', color='cyan', edgecolor='black', alpha=0.8)
            all_bars = [bars_cnn, bars_hyb, bars_stat]

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

    # --- Bottom: AI vs Statistics weight profile ---
    ax3.fill_between(x, 0, weights_ai, alpha=0.3, color='red', label='AI Weight')
    ax3.fill_between(x, weights_ai, 1, alpha=0.3, color='cyan', label='Statistics Weight')
    ax3.plot(x, weights_ai, 'r-', linewidth=1.0, alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Weight', fontsize=12)
    ax3.set_title(f'AI vs Statistics Weight (CV threshold = {cv_threshold})',
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, f'Prediction_{date_str.replace(":", "-")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to: {plot_path}")

    # 10. Print KPIs to console as well
    if has_actual and cnn_m is not None:
        print(f"\nCNN-LSTM  | RMSE: {cnn_m['rmse']:.2f} kW, "
              f"MAE: {cnn_m['mae']:.2f} kW, MAPE: {cnn_m['mape']:.2f}%")
        if hybrid_m is not None:
            print(f"Hybrid    | RMSE: {hybrid_m['rmse']:.2f} kW, "
                  f"MAE: {hybrid_m['mae']:.2f} kW, MAPE: {hybrid_m['mape']:.2f}%")
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
        'Hybrid_Load_kW': hybrid_pred,
        'Statistical_Load_kW': stat_values,
        'CV': cv_values,
        'AI_Weight': weights_ai,
    })
    csv_path = os.path.join(SCRIPT_DIR, f'Prediction_{date_str.replace(":", "-")}.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")

    # Summary
    print(f"\n--- Prediction Summary ---")
    print(f"  Period: {future_ts[0]} to {future_ts[-1]}")
    print(f"  CNN-LSTM  -> Mean: {y_pred.mean():.1f} kW, "
          f"Min: {y_pred.min():.1f}, Max: {y_pred.max():.1f}")
    print(f"  Hybrid    -> Mean: {hybrid_pred.mean():.1f} kW, "
          f"Min: {hybrid_pred.min():.1f}, Max: {hybrid_pred.max():.1f}")
    print(f"  Stats     -> Mean: {np.nanmean(stat_values):.1f} kW")
    print(f"  Avg AI weight: {weights_ai.mean():.2f} (0=all stats, 1=all AI)")


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
  python CNN_LSTM_Prediction.py --predict 2026-02-15 --cv-threshold 0.3
  python CNN_LSTM_Prediction.py --predict 2026-02-15 --n-weeks 6 --agg-method median
""")
    parser.add_argument('--train', action='store_true',
                        help='Train the model on historical data')
    parser.add_argument('--predict', type=str, metavar='DATE',
                        help='Predict 48h from DATE (format: YYYY-MM-DD)')
    parser.add_argument('--cv-threshold', type=float, default=HYBRID_CV_THRESHOLD,
                        help=f'CV threshold for hybrid blending (default: {HYBRID_CV_THRESHOLD})')
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
                    cv_threshold=args.cv_threshold,
                    n_weeks=args.n_weeks,
                    agg_method=args.agg_method)
    else:
        parser.print_help()
