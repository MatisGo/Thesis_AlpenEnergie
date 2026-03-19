"""
DNN_Prod_Prediction.py
======================
Simple Dense Neural Network to predict the next day's total hydro production.

Target  : sum of Production_Is over all 5-min steps of the next calendar day
Unit    : kW  (daily average power)

Features (28 total):
  - Last 20 daily production totals          (lag features)
  - sin/cos of day-of-year                   (2 seasonal features)
  - sin/cos of month                         (2 seasonal features)
  - Daily rain forecast for target day       (from Imported_Forecast.xlsx)
  - Daily mean temperature for target day    (from Imported_Forecast.xlsx)
  - Last Bidmi reservoir height              (from Data_Prediction.xlsx)
  - Last Haselholtz reservoir height         (from Data_Prediction.xlsx)

Commands:
  python DNN_Prod_Prediction.py --train
  python DNN_Prod_Prediction.py --predict
"""

import os
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

# =============================================================================
# CONFIGURATION  — change model structure here
# =============================================================================

N_LAG_DAYS    = 20          # how many past daily production values to use
HIDDEN_LAYERS = [256, 128, 128, 64, 32]  # neurons per hidden layer — add/remove freely
DROPOUT_RATE  = 0.05         # dropout after each hidden layer (0 = no dropout)
EPOCHS        = 500
BATCH_SIZE    = 16
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.15        # fraction of training data used for validation
RANDOM_SEED   = 42

# Columns in Data_Prediction.xlsx
COL_TIME       = 'Time '
COL_PROD       = 'Production_Is'
COL_BIDMI      = 'Level of Bidmi'
COL_HASEL      = 'Level of haselholz'

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(SCRIPT_DIR, 'Ressource', 'Data_Prediction.xlsx')
WEATHER_PATH  = os.path.join(SCRIPT_DIR, 'Ressource', 'Imported_Forecast.xlsx')
MODEL_DIR     = os.path.join(SCRIPT_DIR, 'Model')
MODEL_PATH    = os.path.join(MODEL_DIR, 'DNN_Prod_Model.keras')
SCALER_PATH   = os.path.join(MODEL_DIR, 'DNN_Prod_Scaler.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_daily_production() -> pd.DataFrame:
    """
    Load Data_Prediction.xlsx and return a daily DataFrame with:
      - prod_total   : daily average of Production_Is (kW)
      - bidmi_last   : last Bidmi height of the day
      - hasel_last   : last Haselholtz height of the day
    """
    print("Loading Data_Prediction.xlsx ...")
    df_raw = pd.read_excel(DATA_PATH, sheet_name='Data_January', header=0)

    # Drop the two header rows (Einheit, Signalname)
    df_raw = df_raw.iloc[2:].reset_index(drop=True)

    # Parse timestamp from first column
    df_raw['DateTime'] = pd.to_datetime(
        df_raw.iloc[:, 0].astype(str).str.strip(),
        format='%d.%m.%Y %H:%M:%S', errors='coerce'
    )
    df_raw = df_raw.dropna(subset=['DateTime']).copy()
    df_raw['Date'] = df_raw['DateTime'].dt.date

    # Numeric columns
    df_raw[COL_PROD]  = pd.to_numeric(df_raw[COL_PROD],  errors='coerce')
    df_raw[COL_BIDMI] = pd.to_numeric(df_raw[COL_BIDMI], errors='coerce')
    df_raw[COL_HASEL] = pd.to_numeric(df_raw[COL_HASEL], errors='coerce')

    # Aggregate per day
    daily = df_raw.groupby('Date').agg(
        prod_total = (COL_PROD,  'mean'),
        bidmi_last = (COL_BIDMI, 'last'),
        hasel_last = (COL_HASEL, 'last'),
    ).reset_index()

    # Keep only days with at least 200 valid production readings
    prod_counts = df_raw.groupby('Date')[COL_PROD].count()
    valid_days  = prod_counts[prod_counts >= 200].index
    daily = daily[daily['Date'].isin(valid_days)].copy()

    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)

    print(f"  Daily production rows: {len(daily)}  "
          f"({daily['Date'].iloc[0].date()} → {daily['Date'].iloc[-1].date()})")
    return daily


def load_daily_weather() -> pd.DataFrame:
    """
    Load Imported_Forecast.xlsx and return daily aggregates:
      - rain_sum  : total daily rain (mm)
      - temp_mean : mean daily temperature (°C)
    """
    print("Loading Imported_Forecast.xlsx ...")
    df_w = pd.read_excel(WEATHER_PATH, sheet_name=0, header=0)

    # Expect columns: Time/DateTime, Temperature_C, Rain_Sum_mm (or similar)
    # Normalise column names
    df_w.columns = df_w.columns.str.strip()
    # 'Time' column is 'DD.MM.YYYY HH:MM:SS' — must specify format explicitly
    df_w['DateTime'] = pd.to_datetime(df_w['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df_w = df_w.dropna(subset=['DateTime']).copy()
    df_w['Date'] = df_w['DateTime'].dt.date

    temp_col = [c for c in df_w.columns if 'temp' in c.lower()][0]
    rain_col = [c for c in df_w.columns if 'rain' in c.lower()][0]

    df_w[temp_col] = pd.to_numeric(df_w[temp_col], errors='coerce')
    df_w[rain_col] = pd.to_numeric(df_w[rain_col], errors='coerce')

    daily_w = df_w.groupby('Date').agg(
        rain_sum  = (rain_col,  'sum'),
        temp_mean = (temp_col,  'mean'),
    ).reset_index()

    daily_w['Date'] = pd.to_datetime(daily_w['Date'])
    daily_w = daily_w.sort_values('Date').reset_index(drop=True)

    print(f"  Daily weather rows  : {len(daily_w)}  "
          f"({daily_w['Date'].iloc[0].date()} → {daily_w['Date'].iloc[-1].date()})")
    return daily_w


def merge_and_build_features(daily: pd.DataFrame,
                              daily_w: pd.DataFrame) -> pd.DataFrame:
    """Merge production + weather and add date encoding features."""
    df = daily.merge(daily_w, on='Date', how='left')

    # Date cyclical encoding
    doy = df['Date'].dt.dayofyear
    mon = df['Date'].dt.month
    df['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    df['sin_mon'] = np.sin(2 * np.pi * mon / 12)
    df['cos_mon'] = np.cos(2 * np.pi * mon / 12)

    return df.reset_index(drop=True)


# =============================================================================
# DATASET BUILDER
# =============================================================================

def build_dataset(df: pd.DataFrame):
    """
    Build (X, y, dates) arrays.

    For each target day T+1:
      X = [prod[T-19..T],  sin_doy[T+1], cos_doy[T+1],
            sin_mon[T+1],  cos_mon[T+1],
            rain_sum[T+1], temp_mean[T+1],
            bidmi_last[T], hasel_last[T]]
      y = prod_total[T+1]
    """
    X_list, y_list, dates = [], [], []

    prod  = df['prod_total'].values
    bidmi = df['bidmi_last'].values
    hasel = df['hasel_last'].values
    feats = df[['sin_doy', 'cos_doy', 'sin_mon', 'cos_mon',
                 'rain_sum', 'temp_mean']].values

    for i in range(N_LAG_DAYS, len(df) - 1):
        target_idx = i + 1

        lag    = prod[i - N_LAG_DAYS: i]         # last 20 daily productions
        cal    = feats[target_idx]                # date + weather of target day
        reserv = np.array([bidmi[i], hasel[i]])   # reservoir heights at day T

        x = np.concatenate([lag, cal, reserv])
        X_list.append(x)
        y_list.append(prod[target_idx])
        dates.append(df['Date'].iloc[target_idx])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, dates


# =============================================================================
# MODEL
# =============================================================================

def build_model(input_dim: int):
    """Build a configurable Dense Neural Network."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    tf.random.set_seed(RANDOM_SEED)

    inp = tf.keras.Input(shape=(input_dim,))
    x   = inp
    for units in HIDDEN_LAYERS:
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        if DROPOUT_RATE > 0:
            x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='huber',
        metrics=['mae']
    )
    model.summary()
    return model


# =============================================================================
# TRAINING
# =============================================================================

def run_train():
    import tensorflow as tf
    np.random.seed(RANDOM_SEED)

    daily   = load_daily_production()
    daily_w = load_daily_weather()
    df      = merge_and_build_features(daily, daily_w)

    X, y, dates = build_dataset(df)
    print(f"\nDataset: {len(X)} samples  |  {X.shape[1]} features")

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Save scalers
    joblib.dump({'X': scaler_X, 'y': scaler_y}, SCALER_PATH)

    # Random validation split
    n = len(X_sc)
    val_size  = max(1, int(n * VAL_SPLIT))
    val_idx   = np.random.choice(n, size=val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    X_tr, X_val = X_sc[train_idx], X_sc[val_idx]
    y_tr, y_val = y_sc[train_idx], y_sc[val_idx]

    print(f"Train: {len(X_tr)}  |  Val: {len(X_val)}")

    model = build_model(X.shape[1])

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=60, restore_best_weights=True,
                                         monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=25, factor=0.5, min_lr=1e-6),
    ]

    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=cb, verbose=1)

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'],     label='Train loss')
    ax.plot(history.history['val_loss'], label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title('DNN Production — Training Loss')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 2. Predicted vs Actual on full training set
    y_pred_sc = model.predict(X_sc).ravel()
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    y_actual  = y

    mae  = np.mean(np.abs(y_pred - y_actual))
    mape = np.mean(np.abs((y_pred - y_actual) / (y_actual + 1e-6))) * 100
    rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2))
    print(f"\n--- Training KPIs (full dataset) ---")
    print(f"  RMSE : {rmse:,.0f} kW")
    print(f"  MAE  : {mae:,.0f} kW")
    print(f"  MAPE : {mape:.1f} %")

    dates_dt = [d.date() if hasattr(d, 'date') else d for d in dates]
    fig, ax  = plt.subplots(figsize=(14, 5))
    ax.plot(dates_dt, y_actual / 1e3, label='Actual',    color='steelblue', linewidth=1)
    ax.plot(dates_dt, y_pred   / 1e3, label='Predicted', color='orange',    linewidth=1, linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Average Production (kW)')
    ax.set_title('DNN Production — Actual vs Predicted')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 3. Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_actual / 1e3, y_pred / 1e3, alpha=0.5, s=15, color='steelblue')
    lims = [min(y_actual.min(), y_pred.min()) / 1e3,
            max(y_actual.max(), y_pred.max()) / 1e3]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect forecast')
    ax.set_xlabel('Actual (kW)')
    ax.set_ylabel('Predicted (kW)')
    ax.set_title('Actual vs Predicted — Scatter')
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# PREDICTION
# =============================================================================

def run_predict(show_plots: bool = True):
    from tensorflow.keras.models import load_model

    scalers   = joblib.load(SCALER_PATH)
    scaler_X  = scalers['X']
    scaler_y  = scalers['y']
    model     = load_model(MODEL_PATH)

    daily   = load_daily_production()
    daily_w = load_daily_weather()
    df      = merge_and_build_features(daily, daily_w)

    # T = today (calendar date); use last available production row for lag features
    today_date  = pd.Timestamp(datetime.date.today())
    target_date = today_date + pd.Timedelta(days=1)
    today_idx   = len(df) - 1   # last row with production data (may be yesterday)

    print(f"\n  Reference day (T) : {today_date.date()}")
    print(f"  Predicting    (T+1): {target_date.date()}")

    if today_idx < N_LAG_DAYS:
        print(f"  ERROR: need at least {N_LAG_DAYS} days of production history.")
        return None, None

    # Look up T+1 weather directly from daily_w (not from merged df,
    # since production data for tomorrow does not exist yet)
    target_w = daily_w[daily_w['Date'] == target_date]
    if target_w.empty:
        print(f"\n  WARNING: No weather forecast found for {target_date.date()}.")
        print("  Run get_weather_data.py first to fetch the forecast.")
        return None, None

    # Build date cyclical features for target day
    doy = target_date.dayofyear
    mon = target_date.month
    cal = np.array([
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * mon / 12),
        np.cos(2 * np.pi * mon / 12),
        float(target_w['rain_sum'].iloc[0]),
        float(target_w['temp_mean'].iloc[0]),
    ])

    prod  = df['prod_total'].values
    bidmi = df['bidmi_last'].values
    hasel = df['hasel_last'].values

    lag    = prod[today_idx - N_LAG_DAYS: today_idx]
    reserv = np.array([bidmi[today_idx], hasel[today_idx]])
    x      = np.concatenate([lag, cal, reserv]).reshape(1, -1)

    x_sc      = scaler_X.transform(x)
    y_sc      = model.predict(x_sc, verbose=0).ravel()
    y_pred    = scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()[0]
    print(f"\n{'='*50}")
    print(f"  Predicted production for {target_date.date()}")
    print(f"  {y_pred:>12,.1f}  kW  (daily average)")
    print(f"{'='*50}")

    # Bar chart: last 10 days + prediction
    n_bars     = 10
    hist_start = max(0, today_idx - n_bars + 1)
    hist_df    = df.iloc[hist_start: today_idx + 1]
    bar_dates  = [d.date() for d in hist_df['Date']] + [target_date.date()]
    bar_values = list(hist_df['prod_total'].values) + [y_pred]
    colors     = ['steelblue'] * len(hist_df) + ['orange']

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(bar_dates)), bar_values, color=colors, edgecolor='black', alpha=0.85)
    ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8)
    ax.set_xticks(range(len(bar_dates)))
    ax.set_xticklabels([str(d) for d in bar_dates], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Daily Average Production (kW)')
    ax.set_title(f'Production Forecast — {target_date.date()}  '
                 f'({y_pred:,.0f} kW predicted avg)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='steelblue', label='Historical'),
                        Patch(color='orange',    label='Predicted')])
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    return y_pred, target_date


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DNN Daily Production Forecast',
        epilog="""Examples:
  python DNN_Prod_Prediction.py --train
  python DNN_Prod_Prediction.py --predict"""
    )
    parser.add_argument('--train',   action='store_true', help='Train the DNN model')
    parser.add_argument('--predict', action='store_true', help='Predict next day production')
    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.predict:
        run_predict()
    else:
        parser.print_help()
