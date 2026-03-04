"""
CNN-LSTM 48h Load Prediction
=============================
Hybrid CNN-LSTM model for 48-hour ahead electricity load forecasting.

Architecture (based on Chung & Jang 2022, Khan et al. 2020):
  - 3x Conv1D blocks with aggressive pooling (2016 -> 63 timesteps)
  - 2x LSTM layers for temporal pattern learning
  - Reshaped decoder: RepeatVector + LSTM decoder + TimeDistributed Dense
    outputs (48 hours, 12 steps each) then flattened to 576 steps

Input features (7-day lookback, 2016 steps x 10 features):
  Load_Is (past), Load_yesterday, Load_last_week,
  Hour_sin, Hour_cos, Weekday_sin, Weekday_cos,
  PHolyday, Temp_Forecast, Rain_Forecast

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
                                      Flatten)
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
STEP_SIZE = 6               # Sample every 30 min (reduces memory, keeps enough samples)
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

    X, y, ts, fc = [], [], [], []

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

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    ts = np.array(ts)
    fc = np.array(fc, dtype=np.float32)

    print(f"  Created {len(X)} sequences")
    print(f"  X shape: {X.shape}  (samples, lookback, features)")
    print(f"  y shape: {y.shape}  (samples, forecast_horizon)")

    return X, y, ts, fc


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_model(input_shape, output_hours=OUTPUT_HOURS, steps_per_hour=STEPS_PER_HOUR):
    """
    Build CNN-LSTM hybrid model with decoder architecture.

    Encoder: CNN extracts spatial features (2016 -> 63), LSTM captures temporal patterns.
    Decoder: RepeatVector + LSTM decoder + TimeDistributed Dense produces
             (48 hours, 12 steps/hour) then flattened to 576 steps.

    This structured output helps the model learn the intraday load shape
    rather than producing a smooth average.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, features) = (2016, 10)
    output_hours : int
        Number of hourly blocks = 48
    steps_per_hour : int
        5-min steps per hour = 12

    Returns
    -------
    model : keras.Model
    """
    inputs = Input(shape=input_shape)

    # --- CNN Encoder (3 blocks: 2016 -> 504 -> 126 -> 63) ---
    x = inputs
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
    # Repeat the encoded vector for each output hour
    x = Dense(DENSE_UNITS, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = RepeatVector(output_hours)(x)  # (batch, 48, 256)

    # LSTM decoder processes each hour sequentially
    x = LSTM(128, return_sequences=True)(x)  # (batch, 48, 128)
    x = Dropout(DROPOUT_RATE)(x)

    # TimeDistributed Dense: each hour -> 12 five-min steps
    x = TimeDistributed(Dense(64, activation='relu'))(x)  # (batch, 48, 64)
    x = TimeDistributed(Dense(steps_per_hour))(x)          # (batch, 48, 12)

    # Flatten to (batch, 576)
    outputs = Reshape((output_hours * steps_per_hour,))(x)

    model = Model(inputs=inputs, outputs=outputs)

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

def save_config(scaler_X, scaler_y):
    """Save scalers and model configuration for later prediction."""
    np.savez(CONFIG_PATH,
             lookback_steps=LOOKBACK_STEPS,
             forecast_horizon=FORECAST_HORIZON,
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
             scaler_y_data_range_=scaler_y.data_range_)
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

    return scaler_X, scaler_y


# =============================================================================
# TRAIN MODE
# =============================================================================

def run_train():
    """Train the CNN-LSTM model, evaluate on held-out test set, save model."""
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

    X, y, timestamps, forecast_loads = create_sequences(df, step=STEP_SIZE)

    if len(X) == 0:
        print("ERROR: No valid sequences could be created. Check data.")
        return

    # 3. Train/test split by time
    print("\n--- Step 3: Train/Test Split ---")
    last_date = pd.Timestamp(timestamps[-1])
    test_cutoff = last_date - pd.Timedelta(days=TEST_DAYS)

    mask_train = timestamps < np.datetime64(test_cutoff)
    mask_test = timestamps >= np.datetime64(test_cutoff)

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    ts_test = timestamps[mask_test]
    fc_test = forecast_loads[mask_test]

    print(f"  Test cutoff: {test_cutoff.strftime('%Y-%m-%d')}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Not enough data for train/test split.")
        return

    # 4. Scale features and targets
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

    print(f"  Feature range: [{scaler_X.data_min_.min():.1f}, {scaler_X.data_max_.max():.1f}]")
    print(f"  Target range:  [{scaler_y.data_min_.min():.1f}, {scaler_y.data_max_.max():.1f}]")

    # 5. Build model
    print("\n--- Step 5: Building Model ---")
    model = build_model(input_shape=(n_steps, n_feat))
    model.summary()

    # 6. Train
    print("\n--- Step 6: Training ---")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
    ]

    start_time = time.time()
    history = model.fit(
        X_train_s, y_train_s,
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
    y_pred_s = model.predict(X_test_s, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s)

    # 8. Visualize and compare
    plot_evaluation(y_test, y_pred, fc_test, ts_test, history)

    # 9. Save config
    print("\n--- Step 8: Saving Model & Config ---")
    save_config(scaler_X, scaler_y)
    print(f"  Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


# =============================================================================
# PREDICT MODE
# =============================================================================

def run_predict(date_str):
    """
    Load saved model and predict 48h from a given date.

    Parameters
    ----------
    date_str : str
        Prediction start date in format YYYY-MM-DD or YYYY-MM-DD HH:MM
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
    scaler_X, scaler_y = load_config()
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

    # 5. Extract input (7 days before prediction point)
    input_df = df.iloc[idx - LOOKBACK_STEPS:idx]
    X = input_df[INPUT_FEATURES].values.astype(np.float32)

    if np.isnan(X).any():
        nan_cols = [INPUT_FEATURES[c] for c in range(X.shape[1]) if np.isnan(X[:, c]).any()]
        print(f"WARNING: NaN in input features: {nan_cols}. Filling with interpolation.")
        X = pd.DataFrame(X, columns=INPUT_FEATURES).interpolate().bfill().ffill().values

    # 6. Scale and predict
    X_scaled = scaler_X.transform(X.reshape(-1, n_feat)).reshape(1, LOOKBACK_STEPS, n_feat)
    y_pred_s = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s).flatten()

    # 7. Generate timestamps
    start_time = df['DateTime'].iloc[idx]
    future_ts = [start_time + pd.Timedelta(minutes=5 * i)
                 for i in range(FORECAST_HORIZON)]

    # 8. Get actual and external forecast if available in the dataset
    end_idx = min(idx + FORECAST_HORIZON, len(df))
    future_df = df.iloc[idx:end_idx]
    actual_load = pd.to_numeric(future_df['Load_Is'], errors='coerce').values
    external_load = pd.to_numeric(future_df['Forecast_Load'], errors='coerce').values

    # 9. Compute KPIs and Plot
    has_actual = len(actual_load) > 0 and np.isfinite(actual_load).any()
    ext_valid = np.isfinite(external_load) & (external_load > 0)

    # Compute metrics if actual data is available
    cnn_m = None
    ext_m = None
    if has_actual:
        pred_slice = y_pred[:len(actual_load)]
        cnn_m = compute_metrics(actual_load, pred_slice)
        if ext_valid.any():
            ext_m = compute_metrics(actual_load[ext_valid[:len(actual_load)]],
                                     external_load[ext_valid[:len(external_load)]])

    # Choose layout: 2 panels if we have KPIs, 1 panel if no actual data
    if has_actual and cnn_m is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6),
                                        gridspec_kw={'width_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(16, 6))
        ax2 = None

    # --- Left panel: 3-curve forecast ---
    x = np.arange(FORECAST_HORIZON)
    ax1.plot(x, y_pred, 'r-', label='CNN-LSTM Prediction', linewidth=1.5)

    if has_actual:
        ax1.plot(x[:len(actual_load)], actual_load, 'b-',
                 label='Actual (Load_Is)', linewidth=1.5)

    if ext_valid.any():
        ax1.plot(x[:len(external_load)][ext_valid[:len(external_load)]],
                 external_load[ext_valid[:len(external_load)]],
                 'g-.', label='External Forecast', linewidth=1.5)

    ax1.set_title(f'48h Load Prediction from {date_str}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Load (kW)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Hour tick labels
    tick_pos = np.arange(0, FORECAST_HORIZON, 72)  # every 6 hours
    tick_labels = [(start_time + pd.Timedelta(minutes=5 * p)).strftime('%m-%d %H:%M')
                   for p in tick_pos]
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

    # --- Right panel: KPI bar chart ---
    if ax2 is not None and cnn_m is not None:
        metric_names = ['RMSE (kW)', 'MAE (kW)', 'MAPE (%)']
        cnn_vals = [cnn_m['rmse'], cnn_m['mae'], cnn_m['mape']]

        x_pos = np.arange(len(metric_names))
        width = 0.35

        if ext_m is not None:
            ext_vals = [ext_m['rmse'], ext_m['mae'], ext_m['mape']]
            bars_cnn = ax2.bar(x_pos - width / 2, cnn_vals, width,
                               label='CNN-LSTM', color='steelblue', edgecolor='black', alpha=0.8)
            bars_ext = ax2.bar(x_pos + width / 2, ext_vals, width,
                               label='External', color='coral', edgecolor='black', alpha=0.8)
            for bar in bars_ext:
                h = bar.get_height()
                if np.isfinite(h):
                    ax2.text(bar.get_x() + bar.get_width() / 2, h,
                             f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            bars_cnn = ax2.bar(x_pos, cnn_vals, width,
                               label='CNN-LSTM', color='steelblue', edgecolor='black', alpha=0.8)

        for bar in bars_cnn:
            h = bar.get_height()
            if np.isfinite(h):
                ax2.text(bar.get_x() + bar.get_width() / 2, h,
                         f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metric_names, fontsize=11)
        ax2.set_title('Prediction KPIs', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, f'Prediction_{date_str.replace(":", "-")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to: {plot_path}")

    # 10. Print KPIs to console as well
    if has_actual and cnn_m is not None:
        print(f"\nCNN-LSTM  | RMSE: {cnn_m['rmse']:.2f} kW, "
              f"MAE: {cnn_m['mae']:.2f} kW, MAPE: {cnn_m['mape']:.2f}%")
        if ext_m is not None:
            print(f"External  | RMSE: {ext_m['rmse']:.2f} kW, "
                  f"MAE: {ext_m['mae']:.2f} kW, MAPE: {ext_m['mape']:.2f}%")
    else:
        print("\nNo actual load data available for this period (future prediction).")

    # 11. Save predictions to CSV
    pred_df = pd.DataFrame({
        'DateTime': future_ts,
        'Predicted_Load_kW': y_pred,
    })
    csv_path = os.path.join(SCRIPT_DIR, f'Prediction_{date_str.replace(":", "-")}.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")

    # Summary
    print(f"\n--- Prediction Summary ---")
    print(f"  Period: {future_ts[0]} to {future_ts[-1]}")
    print(f"  Mean predicted load: {y_pred.mean():.1f} kW")
    print(f"  Min:  {y_pred.min():.1f} kW")
    print(f"  Max:  {y_pred.max():.1f} kW")


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
""")
    parser.add_argument('--train', action='store_true',
                        help='Train the model on historical data')
    parser.add_argument('--predict', type=str, metavar='DATE',
                        help='Predict 48h from DATE (format: YYYY-MM-DD)')

    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.predict:
        run_predict(args.predict)
    else:
        parser.print_help()
