"""
DNN_Load_Baseline.py
====================
Simple TimeDistributed MLP baseline for 48h load forecasting.
Designed for direct comparison against CNN_LSTM_Prediction_TEST.py.

Architecture
------------
For each of the 576 forecast steps the model receives only DECODER-stage
features — no raw historical lookback (this is the key architectural
difference vs the CNN-LSTM):

  Per-step input  (N_FEATURES = STAT_N_WEEKS + 10):
    Stat profiles   — STAT_N_WEEKS past same-weekday load values
    PV estimate     — 1 value
    Weather         — Temp, Irradiance, Rain, Cloud  (4 values)
    Time encodings  — Hour_sin/cos, Weekday_sin/cos  (4 values)
    Public holiday  — 1 value

  Input(576, N_FEATURES)
  → TimeDistributed Dense(UNITS_1, relu) → Dropout
  → TimeDistributed Dense(UNITS_2, relu) → Dropout
  → TimeDistributed Dense(1)
  → output (576,)

Hyperparameters to tune (top of file):
  UNITS_1, UNITS_2, DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, PATIENCE

Usage
-----
  python DNN_Load_Baseline.py --train
"""

import argparse
import os
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'Output Forecast')
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'Model', 'DNN_Load_Model_48h.keras')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'Model', 'DNN_Load_Config_48h.npz')
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Shared data pipeline (guarantees identical data & splits vs CNN-LSTM)
# ---------------------------------------------------------------------------
from CNN_LSTM_Prediction_TEST import (
    FORECAST_HORIZON, LOOKBACK_STEPS, STAT_N_WEEKS,
    STEP_SIZE, TEST_DAYS,
    build_stat_profiles, create_sequences,
    load_data, load_pv_nn_model,
)

# ===========================================================================
# HYPERPARAMETERS  ← change these for sensitivity analysis
# ===========================================================================
UNITS_1       = 128    # neurons in first hidden layer
UNITS_2       = 64     # neurons in second hidden layer
DROPOUT_RATE  = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE    = 32
EPOCHS        = 100
PATIENCE      = 15
VAL_RATIO     = 0.15   # fraction of training data kept for validation

# ---------------------------------------------------------------------------
# Per-step feature layout (derived — do not edit)
# ---------------------------------------------------------------------------
WEATHER_COLS = ['Temp_Forecast', 'Irr_FC', 'Rain_Forecast', 'Cloud_Cover']
TIME_COLS    = ['Hour_sin', 'Hour_cos', 'Weekday_sin', 'Weekday_cos']
N_FEATURES   = STAT_N_WEEKS + 1 + len(WEATHER_COLS) + len(TIME_COLS) + 1


# ===========================================================================
# DATA
# ===========================================================================

def build_dnn_sequences(df, start_indices, stat_profiles, pv_horizons):
    """
    Build the per-step input tensor X  shape (N, FORECAST_HORIZON, N_FEATURES).

    Column layout per step:
      [0 : STAT_N_WEEKS]          stat profiles (8 past same-weekday values)
      [STAT_N_WEEKS]               PV estimate
      [STAT_N_WEEKS+1 : +5]       weather (Temp, Irr, Rain, Cloud)
      [STAT_N_WEEKS+5 : +9]       time encodings (Hour_sin/cos, Weekday_sin/cos)
      [STAT_N_WEEKS+9]            public holiday flag
    """
    weather = df[WEATHER_COLS].ffill().fillna(0).values.astype(np.float32)
    time_   = df[TIME_COLS].values.astype(np.float32)
    holiday = df['PHolyday'].fillna(0).values.astype(np.float32)

    N = len(start_indices)
    X = np.zeros((N, FORECAST_HORIZON, N_FEATURES), dtype=np.float32)

    s0 = STAT_N_WEEKS
    for s, idx in enumerate(start_indices):
        fc_s = idx + LOOKBACK_STEPS
        fc_e = fc_s + FORECAST_HORIZON
        if fc_e > len(df):
            continue
        X[s, :, :s0]      = stat_profiles[s]           # (576, STAT_N_WEEKS)
        X[s, :, s0]       = pv_horizons[s]             # (576,)
        X[s, :, s0+1:s0+5] = weather[fc_s:fc_e]        # (576, 4)
        X[s, :, s0+5:s0+9] = time_[fc_s:fc_e]          # (576, 4)
        X[s, :, s0+9]     = holiday[fc_s:fc_e]          # (576,)
    return X


# ===========================================================================
# MODEL
# ===========================================================================

def build_model():
    inp = Input(shape=(FORECAST_HORIZON, N_FEATURES), name='decoder_input')
    x   = TimeDistributed(Dense(UNITS_1, activation='relu'))(inp)
    x   = Dropout(DROPOUT_RATE)(x)
    x   = TimeDistributed(Dense(UNITS_2, activation='relu'))(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = TimeDistributed(Dense(1))(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LEARNING_RATE), loss='mse')
    return model


# ===========================================================================
# TRAINING
# ===========================================================================

def run_train():
    import tensorflow as tf
    np.random.seed(42)
    tf.random.set_seed(42)

    print('=' * 70)
    print('  SIMPLE DNN LOAD BASELINE — TRAINING')
    print(f'  Architecture : Dense({UNITS_1}) → Dense({UNITS_2}) → Dense(1)  '
          f'[TimeDistributed, dropout={DROPOUT_RATE}]')
    print(f'  LR={LEARNING_RATE}  Batch={BATCH_SIZE}  Patience={PATIENCE}')
    print('=' * 70)

    # 1. Load data
    print('\n--- Step 1: Loading Data ---')
    pv_nn = load_pv_nn_model()
    df    = load_data(pv_nn=pv_nn)

    # 2. Create sequences (reuse CNN-LSTM function → same windows & targets)
    print('\n--- Step 2: Creating Sequences ---')
    _, y_raw, timestamps, _, start_indices, pv_horizons = create_sequences(
        df, step=STEP_SIZE)

    # 3. Statistical profiles (same as CNN-LSTM decoder input)
    print('\n--- Step 3: Building Stat Profiles ---')
    stat_profiles = build_stat_profiles(
        df['Load_Is'].values.astype(np.float32),
        start_indices, horizon=FORECAST_HORIZON, n_weeks=STAT_N_WEEKS)

    # 4. Build DNN feature tensor
    print('\n--- Step 4: Building Feature Tensor ---')
    X_all = build_dnn_sequences(df, start_indices, stat_profiles, pv_horizons)
    print(f'  Shape: {X_all.shape}   '
          f'({N_FEATURES} features/step: '
          f'stat×{STAT_N_WEEKS} + PV + weather×4 + time×4 + holiday)')

    # 5. Train / test split — identical cutoff to CNN-LSTM
    print('\n--- Step 5: Train / Test Split ---')
    test_cutoff = pd.Timestamp(timestamps[-1]) - pd.Timedelta(days=TEST_DAYS)
    mask_train  = timestamps < np.datetime64(test_cutoff)
    mask_test   = timestamps >= np.datetime64(test_cutoff)

    X_train, y_train = X_all[mask_train],  y_raw[mask_train]
    X_test,  y_test  = X_all[mask_test],   y_raw[mask_test]
    ts_test          = timestamps[mask_test]
    print(f'  Train: {mask_train.sum():,}  |  Test: {mask_test.sum():,} samples')

    # 6. Scaling
    print('\n--- Step 6: Scaling ---')
    n_tr = len(X_train)

    scaler_X = MinMaxScaler()
    X_tr_sc  = scaler_X.fit_transform(
        X_train.reshape(-1, N_FEATURES)).reshape(n_tr, FORECAST_HORIZON, N_FEATURES)
    X_te_sc  = scaler_X.transform(
        X_test.reshape(-1, N_FEATURES)).reshape(len(X_test), FORECAST_HORIZON, N_FEATURES)

    scaler_y = MinMaxScaler()
    y_tr_sc  = scaler_y.fit_transform(
        y_train.reshape(-1, 1)).reshape(n_tr, FORECAST_HORIZON, 1)
    y_te_sc  = scaler_y.transform(
        y_test.reshape(-1, 1)).reshape(len(y_test), FORECAST_HORIZON, 1)

    # 7. Random validation split (same approach as CNN-LSTM)
    val_size = max(1, int(n_tr * VAL_RATIO))
    val_idx  = np.random.choice(n_tr, size=val_size, replace=False)
    trn_idx  = np.setdiff1d(np.arange(n_tr), val_idx)
    print(f'  Sub-train: {len(trn_idx):,}  |  Val: {len(val_idx):,}')

    # 8. Build & train model
    print('\n--- Step 7: Training ---')
    model = build_model()
    model.summary()

    history = model.fit(
        X_tr_sc[trn_idx], y_tr_sc[trn_idx],
        validation_data=(X_tr_sc[val_idx], y_tr_sc[val_idx]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=PATIENCE,
                          restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
        ],
        verbose=1,
    )
    print(f'\n  Model saved to: {MODEL_PATH}')

    # 9. Evaluate
    print('\n--- Step 8: Evaluation ---')
    y_pred_sc = model.predict(X_te_sc, verbose=0)          # (N, 576, 1)
    y_pred    = scaler_y.inverse_transform(
        y_pred_sc.reshape(-1, 1)).reshape(len(y_test), FORECAST_HORIZON)

    rmse, mae, mape = _compute_kpis(y_pred, y_test)
    print(f'\n  RMSE : {rmse:.1f} kW')
    print(f'  MAE  : {mae:.1f} kW')
    print(f'  MAPE : {mape:.1f} %')

    # 10. Save scalers
    np.savez(CONFIG_PATH,
             scaler_X_min=scaler_X.data_min_,
             scaler_X_scale=scaler_X.scale_,
             scaler_y_min=scaler_y.data_min_,
             scaler_y_scale=scaler_y.scale_,
             n_features=N_FEATURES,
             units_1=UNITS_1, units_2=UNITS_2,
             dropout=DROPOUT_RATE, lr=LEARNING_RATE)
    print(f'  Config saved to: {CONFIG_PATH}')

    # 11. Plots
    _plot_results(history, y_pred, y_test, ts_test, rmse, mae, mape)


# ===========================================================================
# EVALUATION HELPERS
# ===========================================================================

def _compute_kpis(y_pred, y_true):
    pf = y_pred.ravel()
    tf_ = y_true.ravel()
    nz  = tf_ > 50
    rmse = np.sqrt(np.mean((pf - tf_) ** 2))
    mae  = np.mean(np.abs(pf - tf_))
    mape = np.mean(np.abs((pf[nz] - tf_[nz]) / tf_[nz])) * 100
    return rmse, mae, mape


def _plot_results(history, y_pred, y_true, timestamps, rmse, mae, mape):
    ts = pd.DatetimeIndex(timestamps)

    # Use the last sample for the forecast curve
    t_fc   = pd.date_range(start=ts[-1], periods=FORECAST_HORIZON, freq='5min')
    p_fc   = y_pred[-1]
    a_fc   = y_true[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Simple DNN Baseline — Load Forecast Evaluation',
                 fontsize=12, fontweight='bold')

    # Forecast curve
    ax = axes[0, 0]
    ax.plot(t_fc, a_fc, color='#1f77b4', lw=1.5, label='Actual (Load_Is)')
    ax.plot(t_fc, p_fc, color='#d62728', lw=1.5, ls='--', label='DNN Prediction')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_title(f'48h Forecast from {t_fc[0].strftime("%Y-%m-%d %H:%M")}')
    ax.set_ylabel('Load (kW)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    # KPI bars
    ax = axes[0, 1]
    metrics = ['RMSE (kW)', 'MAE (kW)', 'MAPE (%)']
    values  = [rmse, mae, mape]
    bars = ax.bar(metrics, values, color='#4878CF', width=0.45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    ax.set_title('KPIs — Full Test Set')
    ax.set_ylabel('Error')
    ax.grid(True, axis='y', alpha=0.3)

    # Training history
    ax = axes[1, 0]
    ax.plot(history.history['loss'],     label='Training Loss',   color='#1f77b4', lw=1.5)
    ax.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', lw=1.5)
    ax.set_title('Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scatter
    ax = axes[1, 1]
    pf = y_pred.ravel()
    tf_ = y_true.ravel()
    ax.scatter(tf_, pf, s=1, alpha=0.2, color='#4878CF')
    lim = [0, max(tf_.max(), pf.max()) * 1.05]
    ax.plot(lim, lim, 'r--', lw=1, label='Perfect Prediction')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual Load (kW)')
    ax.set_ylabel('Predicted Load (kW)')
    ax.set_title('Actual vs Predicted')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'DNN_Results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n  Results plot saved to: {save_path}')
    plt.show()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simple DNN load forecast baseline',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='Example:\n  python DNN_Load_Baseline.py --train')
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    if args.train:
        run_train()
    else:
        parser.print_help()