"""
DNN_Prod_Prediction.py
======================
Two-stage hydro production forecast:

  Stage 1 — Daily average model (DNN, 1 output):
    Predicts the daily average production (kW) for the next day.

  Stage 2 — Shape model (DNN, 96 outputs):
    Predicts the normalized intraday distribution (96 × 15-min slots).
    Shape is scaled by Stage-1 average, clipped at MAX_PRODUCTION_KW,
    then interpolated to 5-min resolution (288 steps).

Weekend variant (run on Friday):
    Predicts Fri/Sat/Sun/Mon daily averages → shape for each day.
    Only Sat/Sun/Mon written to output.

Commands:
  python DNN_Prod_Prediction.py --train
  python DNN_Prod_Prediction.py --train-weekend
  python DNN_Prod_Prediction.py --train-shape
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
# CONFIGURATION
# =============================================================================

# --- Daily / Weekend average models ---
N_LAG_DAYS            = 20
HIDDEN_LAYERS         = [256, 128, 128, 64, 32]
WEEKEND_HIDDEN_LAYERS = [256, 128, 128, 64, 32]

# --- Shape model ---
N_LAG_DAYS_SHAPE     = 14       # days of 15-min history used as input
SHAPE_STEPS_PER_DAY  = 96       # 15-min slots per day
SHAPE_HIDDEN_LAYERS  = [512, 256, 128]
MAX_PRODUCTION_KW    = 3076     # hard clip after scaling

# --- Common ---
DROPOUT_RATE  = 0.05
EPOCHS        = 500
BATCH_SIZE    = 16
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.15
RANDOM_SEED   = 42

# Columns in Data_Prediction.xlsx
COL_PROD  = 'Production_Is'
COL_BIDMI = 'Level of Bidmi'
COL_HASEL = 'Level of haselholz'

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, 'Ressource', 'Data_Prediction.xlsx')
WEATHER_PATH = os.path.join(SCRIPT_DIR, 'Ressource', 'Imported_Forecast.xlsx')
MODEL_DIR    = os.path.join(SCRIPT_DIR, 'Model')

MODEL_PATH          = os.path.join(MODEL_DIR, 'DNN_Prod_Model.keras')
SCALER_PATH         = os.path.join(MODEL_DIR, 'DNN_Prod_Scaler.pkl')
WEEKEND_MODEL_PATH  = os.path.join(MODEL_DIR, 'DNN_Prod_Model_weekend.keras')
WEEKEND_SCALER_PATH = os.path.join(MODEL_DIR, 'DNN_Prod_Scaler_weekend.pkl')
SHAPE_MODEL_PATH    = os.path.join(MODEL_DIR, 'DNN_Prod_Shape_Model.keras')
SHAPE_SCALER_PATH   = os.path.join(MODEL_DIR, 'DNN_Prod_Shape_Scaler.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def _parse_raw_excel():
    """Parse Data_Prediction.xlsx → 5-min DataFrame with DateTime, prod, bidmi, hasel."""
    df_raw = pd.read_excel(DATA_PATH, sheet_name='Data_January', header=0)
    df_raw = df_raw.iloc[2:].reset_index(drop=True)
    df_raw['DateTime'] = pd.to_datetime(
        df_raw.iloc[:, 0].astype(str).str.strip(),
        format='%d.%m.%Y %H:%M:%S', errors='coerce'
    )
    df_raw = df_raw.dropna(subset=['DateTime']).copy()
    for col in [COL_PROD, COL_BIDMI, COL_HASEL]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    return df_raw


def load_daily_production() -> pd.DataFrame:
    """Return daily DataFrame: Date, prod_total (mean kW), bidmi_last, hasel_last."""
    print("Loading Data_Prediction.xlsx (daily) ...")
    df_raw = _parse_raw_excel()
    df_raw['Date'] = df_raw['DateTime'].dt.date

    daily = df_raw.groupby('Date').agg(
        prod_total = (COL_PROD,  'mean'),
        bidmi_last = (COL_BIDMI, 'last'),
        hasel_last = (COL_HASEL, 'last'),
    ).reset_index()

    prod_counts = df_raw.groupby('Date')[COL_PROD].count()
    valid_days  = prod_counts[prod_counts >= 200].index
    daily = daily[daily['Date'].isin(valid_days)].copy()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)

    print(f"  Daily rows: {len(daily)}  "
          f"({daily['Date'].iloc[0].date()} → {daily['Date'].iloc[-1].date()})")
    return daily


def load_15min_production() -> tuple:
    """
    Return (df_15min, daily_reservoir).
      df_15min        : DateTime (15-min), Date, prod_15min (kW mean)
      daily_reservoir : Date, bidmi_last, hasel_last
    """
    print("Loading Data_Prediction.xlsx (15-min) ...")
    df_raw = _parse_raw_excel()
    df_raw = df_raw.set_index('DateTime').sort_index()

    # Resample to 15-min
    prod_15 = df_raw[COL_PROD].resample('15min').mean()
    df_15   = prod_15.reset_index()
    df_15.columns = ['DateTime', 'prod_15min']
    df_15['Date'] = df_15['DateTime'].dt.date

    # Keep only dates with ≥ 80 valid 15-min slots (full days)
    counts    = df_15.groupby('Date')['prod_15min'].count()
    valid_days = counts[counts >= 80].index
    df_15 = df_15[df_15['Date'].isin(valid_days)].copy()
    df_15['prod_15min'] = df_15['prod_15min'].fillna(0.0)

    # Daily reservoir heights (reuse daily loader result)
    daily     = load_daily_production()
    daily_res = daily[['Date', 'bidmi_last', 'hasel_last']].copy()
    daily_res['Date'] = pd.to_datetime(daily_res['Date'])

    print(f"  15-min rows  : {len(df_15)}  |  unique days: {df_15['Date'].nunique()}")
    return df_15, daily_res


def load_daily_weather() -> pd.DataFrame:
    """Return daily DataFrame: Date, rain_sum, temp_mean."""
    print("Loading Imported_Forecast.xlsx ...")
    df_w = pd.read_excel(WEATHER_PATH, sheet_name=0, header=0)
    df_w.columns = df_w.columns.str.strip()
    df_w['DateTime'] = pd.to_datetime(df_w['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df_w = df_w.dropna(subset=['DateTime']).copy()
    df_w['Date'] = df_w['DateTime'].dt.date

    temp_col = [c for c in df_w.columns if 'temp' in c.lower()][0]
    rain_col = [c for c in df_w.columns if 'rain' in c.lower()][0]
    df_w[temp_col] = pd.to_numeric(df_w[temp_col], errors='coerce')
    df_w[rain_col] = pd.to_numeric(df_w[rain_col], errors='coerce')

    daily_w = df_w.groupby('Date').agg(
        rain_sum  = (rain_col, 'sum'),
        temp_mean = (temp_col, 'mean'),
    ).reset_index()
    daily_w['Date'] = pd.to_datetime(daily_w['Date'])
    return daily_w.sort_values('Date').reset_index(drop=True)


def _weather_lookup(daily_w: pd.DataFrame) -> dict:
    return {row['Date'].date(): (row['rain_sum'], row['temp_mean'])
            for _, row in daily_w.iterrows()}


def merge_and_build_features(daily, daily_w):
    df = daily.merge(daily_w, on='Date', how='left')
    doy = df['Date'].dt.dayofyear
    mon = df['Date'].dt.month
    df['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    df['sin_mon'] = np.sin(2 * np.pi * mon / 12)
    df['cos_mon'] = np.cos(2 * np.pi * mon / 12)
    return df.reset_index(drop=True)


# =============================================================================
# DATASET BUILDERS
# =============================================================================

def build_dataset(df):
    """Daily model dataset: X → 1 value (next-day avg kW)."""
    X_list, y_list, dates = [], [], []
    prod  = df['prod_total'].values
    bidmi = df['bidmi_last'].values
    hasel = df['hasel_last'].values
    feats = df[['sin_doy', 'cos_doy', 'sin_mon', 'cos_mon',
                 'rain_sum', 'temp_mean']].values

    for i in range(N_LAG_DAYS, len(df) - 1):
        t = i + 1
        lag    = prod[i - N_LAG_DAYS: i]
        cal    = feats[t]
        reserv = np.array([bidmi[i], hasel[i]])
        X_list.append(np.concatenate([lag, cal, reserv]))
        y_list.append(prod[t])
        dates.append(df['Date'].iloc[t])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), dates


def build_dataset_weekend(df, daily_w):
    """
    Weekend dataset: for each Friday → X (34 features), y (4 outputs: Fri/Sat/Sun/Mon avg kW).
    """
    X_list, y_list, dates = [], [], []
    w_dict = _weather_lookup(daily_w)
    prod   = df['prod_total'].values
    bidmi  = df['bidmi_last'].values
    hasel  = df['hasel_last'].values
    date_to_idx = {row['Date'].date(): idx for idx, row in df.iterrows()}

    for i in range(N_LAG_DAYS, len(df)):
        fri_date = df['Date'].iloc[i]
        if fri_date.weekday() != 4:
            continue
        sat_d = (fri_date + pd.Timedelta(days=1)).date()
        sun_d = (fri_date + pd.Timedelta(days=2)).date()
        mon_d = (fri_date + pd.Timedelta(days=3)).date()
        fri_d = fri_date.date()
        thu_d = (fri_date - pd.Timedelta(days=1)).date()

        if not all(d in date_to_idx for d in [fri_d, sat_d, sun_d, mon_d]):
            continue
        if not all(d in w_dict for d in [fri_d, sat_d, sun_d, mon_d]):
            continue

        thu_idx = date_to_idx.get(thu_d, i - 1)
        lag     = prod[i - N_LAG_DAYS: i]
        doy     = fri_date.dayofyear
        mon     = fri_date.month
        cal_date = np.array([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                              np.sin(2*np.pi*mon/12),     np.cos(2*np.pi*mon/12)])
        weather = np.array([
            w_dict[fri_d][0],  w_dict[fri_d][1],
            w_dict[sat_d][0],  w_dict[sat_d][1],
            w_dict[sun_d][0],  w_dict[sun_d][1],
            w_dict[mon_d][0],  w_dict[mon_d][1],
        ], dtype=np.float32)
        reserv = np.array([bidmi[thu_idx], hasel[thu_idx]])

        X_list.append(np.concatenate([lag, cal_date, weather, reserv]))
        y_list.append(np.array([prod[date_to_idx[fri_d]], prod[date_to_idx[sat_d]],
                                 prod[date_to_idx[sun_d]], prod[date_to_idx[mon_d]]],
                                dtype=np.float32))
        dates.append(fri_date)

    if not X_list:
        raise ValueError("No Friday samples with complete data found.")
    print(f"  Weekend samples (Fridays): {len(X_list)}")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), dates


def build_dataset_shape(df_15min, daily_reservoir):
    """
    Shape model dataset.
    X = [normalized_profiles for last 14 days (14×96=1344),
         sin_dow, cos_dow,     (day-of-week of target day)
         bidmi_last, hasel_last]   → 1348 features
    y = normalized 96-slot profile of target day (mean=1.0)
    """
    X_list, y_list, dates = [], [], []

    dates_available = sorted(df_15min['Date'].unique())
    res_lookup = {row['Date'].date(): (row['bidmi_last'], row['hasel_last'])
                  for _, row in daily_reservoir.iterrows()}

    def _get_profile(date_val):
        """Return 96-slot array for a given date, or None if incomplete."""
        slots = df_15min[df_15min['Date'] == date_val]['prod_15min'].values
        if len(slots) < SHAPE_STEPS_PER_DAY:
            return None
        return slots[:SHAPE_STEPS_PER_DAY].astype(np.float32)

    for i in range(N_LAG_DAYS_SHAPE, len(dates_available) - 1):
        target_d = dates_available[i + 1]
        prev_d   = dates_available[i]

        target_prof = _get_profile(target_d)
        if target_prof is None:
            continue
        target_mean = target_prof.mean()
        if target_mean < 50:          # skip near-zero days (shutdowns)
            continue
        y = target_prof / target_mean  # normalized shape

        lag_dates = dates_available[i - N_LAG_DAYS_SHAPE + 1: i + 1]
        if len(lag_dates) != N_LAG_DAYS_SHAPE:
            continue

        lag_profiles = []
        valid = True
        for d in lag_dates:
            p = _get_profile(d)
            if p is None:
                valid = False; break
            d_mean = max(p.mean(), 1.0)
            lag_profiles.append(p / d_mean)
        if not valid:
            continue

        lag_flat = np.concatenate(lag_profiles)   # 1344 values

        dow = pd.Timestamp(target_d).dayofweek
        dow_feat = np.array([np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])

        res = res_lookup.get(prev_d, (0.0, 0.0))
        reserv = np.array(res, dtype=np.float32)

        X_list.append(np.concatenate([lag_flat, dow_feat, reserv]))
        y_list.append(y)
        dates.append(target_d)

    if not X_list:
        raise ValueError("No valid shape training samples found.")
    print(f"  Shape samples: {len(X_list)}  |  features: {len(X_list[0])}")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), dates


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def _build_dense_model(input_dim, output_dim, hidden_layers, name="DNN"):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    tf.random.set_seed(RANDOM_SEED)

    inp = tf.keras.Input(shape=(input_dim,))
    x   = inp
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        if DROPOUT_RATE > 0:
            x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(output_dim, activation='linear')(x)

    model = tf.keras.Model(inp, out, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='huber', metrics=['mae'])
    model.summary()
    return model


def build_model(input_dim):
    return _build_dense_model(input_dim, 1,  HIDDEN_LAYERS,         "DNN_Prod_Daily")

def build_model_weekend(input_dim):
    return _build_dense_model(input_dim, 4,  WEEKEND_HIDDEN_LAYERS, "DNN_Prod_Weekend")

def build_model_shape(input_dim):
    return _build_dense_model(input_dim, SHAPE_STEPS_PER_DAY, SHAPE_HIDDEN_LAYERS, "DNN_Prod_Shape")


# =============================================================================
# COMMON TRAINING HELPER
# =============================================================================

def _fit_model(model, X_sc, y_sc, model_path):
    import tensorflow as tf
    n = len(X_sc)
    val_idx   = np.random.choice(n, size=max(1, int(n * VAL_SPLIT)), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    print(f"Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=60, restore_best_weights=True,
                                         monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=25, factor=0.5, min_lr=1e-6),
    ]
    history = model.fit(X_sc[train_idx], y_sc[train_idx],
                        validation_data=(X_sc[val_idx], y_sc[val_idx]),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=cb, verbose=1)
    model.save(model_path)
    print(f"Model saved: {model_path}")
    return history


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

    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    joblib.dump({'X': scaler_X, 'y': scaler_y}, SCALER_PATH)

    model   = build_model(X.shape[1])
    history = _fit_model(model, X_sc, y_sc, MODEL_PATH)
    _plot_training(history, model, scaler_y, X_sc, y, dates, "Daily")


def run_train_weekend():
    import tensorflow as tf
    np.random.seed(RANDOM_SEED)
    daily   = load_daily_production()
    daily_w = load_daily_weather()
    df      = merge_and_build_features(daily, daily_w)
    X, y, dates = build_dataset_weekend(df, daily_w)
    print(f"\nWeekend dataset: {len(X)} samples  |  {X.shape[1]} features  |  4 outputs")

    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y)
    joblib.dump({'X': scaler_X, 'y': scaler_y}, WEEKEND_SCALER_PATH)

    model   = build_model_weekend(X.shape[1])
    history = _fit_model(model, X_sc, y_sc, WEEKEND_MODEL_PATH)

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Train'); ax.plot(history.history['val_loss'], label='Val')
    ax.set_title('Weekend Model — Loss'); ax.legend(); plt.tight_layout(); plt.show()

    # Per-day actual vs predicted
    y_pred = scaler_y.inverse_transform(model.predict(X_sc))
    labels = ['Friday', 'Saturday', 'Sunday', 'Monday']
    dates_dt = [d.date() if hasattr(d, 'date') else d for d in dates]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, (ax, label) in enumerate(zip(axes.ravel(), labels)):
        ax.plot(dates_dt, y[:, col], label='Actual',    color='steelblue', lw=1)
        ax.plot(dates_dt, y_pred[:, col], label='Pred', color='orange',    lw=1, ls='--')
        ax.set_title(f"{label}  |  MAE={np.mean(np.abs(y_pred[:,col]-y[:,col])):.0f} kW")
        ax.legend()
    plt.suptitle('Weekend Production — Actual vs Predicted'); plt.tight_layout(); plt.show()


def run_train_shape():
    import tensorflow as tf
    np.random.seed(RANDOM_SEED)
    df_15min, daily_res = load_15min_production()
    X, y, dates = build_dataset_shape(df_15min, daily_res)
    print(f"\nShape dataset: {len(X)} samples  |  {X.shape[1]} features  |  96 outputs")

    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y)    # (n, 96) — each slot scaled independently
    joblib.dump({'X': scaler_X, 'y': scaler_y}, SHAPE_SCALER_PATH)

    model   = build_model_shape(X.shape[1])
    history = _fit_model(model, X_sc, y_sc, SHAPE_MODEL_PATH)

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Train'); ax.plot(history.history['val_loss'], label='Val')
    ax.set_title('Shape Model — Training Loss'); ax.set_xlabel('Epoch'); ax.legend()
    plt.tight_layout(); plt.show()

    # Sample predicted shapes vs actual (last 4 validation-style days)
    y_pred_sc = model.predict(X_sc)
    y_pred    = scaler_y.inverse_transform(y_pred_sc)  # (n, 96) normalized shapes

    # Pick 4 random samples to plot
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(y), size=min(4, len(y)), replace=False)
    fig, axes  = plt.subplots(2, 2, figsize=(14, 8))
    t_15 = np.arange(SHAPE_STEPS_PER_DAY) * 15 / 60  # hours
    for ax, idx in zip(axes.ravel(), sample_idx):
        ax.plot(t_15, y[idx],      label='Actual shape',  color='steelblue', lw=1.5)
        ax.plot(t_15, y_pred[idx], label='Predicted',     color='orange',    lw=1.5, ls='--')
        ax.set_title(f"Date: {dates[idx]}")
        ax.set_xlabel('Hour of day'); ax.set_ylabel('Normalized production')
        ax.legend()
    plt.suptitle('Shape Model — Sample Predictions'); plt.tight_layout(); plt.show()


# =============================================================================
# SHAPE PREDICTION HELPER
# =============================================================================

def _predict_shape_for_day(target_date: pd.Timestamp, daily_avg: float,
                            df_15min: pd.DataFrame, daily_reservoir: pd.DataFrame):
    """
    Predict 5-min production profile for target_date.
    Returns DataFrame with columns: DateTime, Predicted_Prod_kW (288 rows).
    Returns None if shape model not available.
    """
    if not os.path.isfile(SHAPE_MODEL_PATH) or not os.path.isfile(SHAPE_SCALER_PATH):
        return None

    from tensorflow.keras.models import load_model as _load
    scalers = joblib.load(SHAPE_SCALER_PATH)
    model   = _load(SHAPE_MODEL_PATH)

    target_d = target_date.date()
    res_lookup = {row['Date'].date(): (row['bidmi_last'], row['hasel_last'])
                  for _, row in daily_reservoir.iterrows()}

    # Last N_LAG_DAYS_SHAPE days BEFORE target_date
    past_dates = sorted(d for d in df_15min['Date'].unique() if d < target_d)
    past_dates = past_dates[-N_LAG_DAYS_SHAPE:]

    if len(past_dates) < N_LAG_DAYS_SHAPE:
        print(f"  Shape model: not enough history ({len(past_dates)}/{N_LAG_DAYS_SHAPE} days)")
        return None

    lag_profiles = []
    for d in past_dates:
        slots = df_15min[df_15min['Date'] == d]['prod_15min'].values
        if len(slots) < SHAPE_STEPS_PER_DAY:
            return None
        d_mean = max(slots[:SHAPE_STEPS_PER_DAY].mean(), 1.0)
        lag_profiles.append(slots[:SHAPE_STEPS_PER_DAY].astype(np.float32) / d_mean)

    lag_flat = np.concatenate(lag_profiles)

    dow      = target_date.dayofweek
    dow_feat = np.array([np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])

    prev_d = past_dates[-1]
    res    = res_lookup.get(prev_d, (0.0, 0.0))
    reserv = np.array(res, dtype=np.float32)

    x    = np.concatenate([lag_flat, dow_feat, reserv]).reshape(1, -1)
    x_sc = scalers['X'].transform(x)

    shape_sc  = model.predict(x_sc, verbose=0)           # (1, 96)
    shape_norm = scalers['y'].inverse_transform(shape_sc)[0]  # 96 normalized values
    shape_norm = np.clip(shape_norm, 0, None)             # no negative production

    # Scale by daily average and hard-clip
    prod_15min = np.clip(shape_norm * daily_avg, 0, MAX_PRODUCTION_KW)

    # Interpolate 15-min (96 slots) → 5-min (288 steps)
    t_15 = np.arange(SHAPE_STEPS_PER_DAY) * 15.0         # [0, 15, 30, ..., 1425] minutes
    t_5  = np.arange(288) * 5.0                           # [0,  5, 10, ..., 1435] minutes
    prod_5min = np.interp(t_5, t_15, prod_15min)
    prod_5min = np.clip(prod_5min, 0, MAX_PRODUCTION_KW)

    base       = pd.Timestamp(target_d)
    timestamps = [base + pd.Timedelta(minutes=5*i) for i in range(288)]

    return pd.DataFrame({
        'DateTime':          timestamps,
        'Predicted_Prod_kW': np.round(prod_5min, 1),
    })


# =============================================================================
# PREDICTION — DAILY
# =============================================================================

def _run_predict_daily(show_plots=True):
    from tensorflow.keras.models import load_model as _load
    scalers = joblib.load(SCALER_PATH)
    model   = _load(MODEL_PATH)
    daily   = load_daily_production()
    daily_w = load_daily_weather()
    df      = merge_and_build_features(daily, daily_w)
    w_dict  = _weather_lookup(daily_w)

    today_date  = pd.Timestamp(datetime.date.today())
    target_date = today_date + pd.Timedelta(days=1)
    today_idx   = len(df) - 1
    print(f"\n  Reference day (T)  : {today_date.date()}")
    print(f"  Predicting   (T+1) : {target_date.date()}")

    if today_idx < N_LAG_DAYS:
        print(f"  ERROR: need {N_LAG_DAYS} days of history."); return []

    t_d = target_date.date()
    if t_d not in w_dict:
        print(f"  WARNING: No weather for {t_d}. Run get_weather_data.py."); return []

    doy = target_date.dayofyear; mon = target_date.month
    cal = np.array([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                    np.sin(2*np.pi*mon/12),      np.cos(2*np.pi*mon/12),
                    w_dict[t_d][0], w_dict[t_d][1]])
    prod   = df['prod_total'].values
    lag    = prod[today_idx - N_LAG_DAYS: today_idx]
    reserv = np.array([df['bidmi_last'].values[today_idx], df['hasel_last'].values[today_idx]])
    x      = np.concatenate([lag, cal, reserv]).reshape(1, -1)

    y_sc   = model.predict(scalers['X'].transform(x), verbose=0).ravel()
    y_pred = scalers['y'].inverse_transform(y_sc.reshape(-1, 1)).ravel()[0]

    print(f"\n{'='*50}")
    print(f"  Predicted avg production for {target_date.date()}")
    print(f"  {y_pred:>12,.1f}  kW")
    print(f"{'='*50}")

    # Shape distribution
    df_15min, daily_res = load_15min_production()
    dist_df = _predict_shape_for_day(target_date, y_pred, df_15min, daily_res)
    if dist_df is not None:
        print(f"  Shape forecast: {len(dist_df)} × 5-min slots  "
              f"(peak={dist_df['Predicted_Prod_kW'].max():.0f} kW)")

    if show_plots:
        _plot_predict_bar(df, today_idx, [(target_date, y_pred)])
        if dist_df is not None:
            _plot_distribution(dist_df, target_date)

    return [(target_date, y_pred, dist_df)]


# =============================================================================
# PREDICTION — WEEKEND
# =============================================================================

def _run_predict_weekend(show_plots=True):
    from tensorflow.keras.models import load_model as _load
    scalers = joblib.load(WEEKEND_SCALER_PATH)
    model   = _load(WEEKEND_MODEL_PATH)
    daily   = load_daily_production()
    daily_w = load_daily_weather()
    df      = merge_and_build_features(daily, daily_w)
    w_dict  = _weather_lookup(daily_w)

    today_date = pd.Timestamp(datetime.date.today())  # Friday
    fri_d = today_date.date()
    sat_d = (today_date + pd.Timedelta(days=1)).date()
    sun_d = (today_date + pd.Timedelta(days=2)).date()
    mon_d = (today_date + pd.Timedelta(days=3)).date()

    print(f"\n  Friday reference   : {fri_d}")
    print(f"  Predicting Sat-Mon : {sat_d} / {sun_d} / {mon_d}")

    missing = [d for d in [fri_d, sat_d, sun_d, mon_d] if d not in w_dict]
    if missing:
        print(f"  WARNING: No weather for {missing}. Run get_weather_data.py."); return []

    today_idx = len(df) - 1
    if today_idx < N_LAG_DAYS:
        print(f"  ERROR: need {N_LAG_DAYS} days of history."); return []

    prod = df['prod_total'].values
    lag  = prod[today_idx - N_LAG_DAYS: today_idx]
    doy  = today_date.dayofyear; mon = today_date.month
    cal_date = np.array([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                         np.sin(2*np.pi*mon/12),      np.cos(2*np.pi*mon/12)])
    weather = np.array([w_dict[fri_d][0], w_dict[fri_d][1],
                        w_dict[sat_d][0], w_dict[sat_d][1],
                        w_dict[sun_d][0], w_dict[sun_d][1],
                        w_dict[mon_d][0], w_dict[mon_d][1]], dtype=np.float32)
    reserv = np.array([df['bidmi_last'].values[today_idx], df['hasel_last'].values[today_idx]])
    x      = np.concatenate([lag, cal_date, weather, reserv]).reshape(1, -1)

    y_sc   = model.predict(scalers['X'].transform(x), verbose=0)
    y_pred = scalers['y'].inverse_transform(y_sc)[0]  # [fri, sat, sun, mon]
    pred_fri, pred_sat, pred_sun, pred_mon = y_pred

    print(f"\n{'='*55}")
    print(f"  Friday   {fri_d}: {pred_fri:>8,.1f} kW  (internal reference)")
    print(f"  Saturday {sat_d}: {pred_sat:>8,.1f} kW")
    print(f"  Sunday   {sun_d}: {pred_sun:>8,.1f} kW")
    print(f"  Monday   {mon_d}: {pred_mon:>8,.1f} kW")
    print(f"{'='*55}")

    # Shape distribution for Sat/Sun/Mon
    df_15min, daily_res = load_15min_production()
    results = []
    for ts, avg in [(pd.Timestamp(sat_d), pred_sat),
                    (pd.Timestamp(sun_d), pred_sun),
                    (pd.Timestamp(mon_d), pred_mon)]:
        dist = _predict_shape_for_day(ts, avg, df_15min, daily_res)
        if dist is not None:
            print(f"  Shape for {ts.date()}: peak={dist['Predicted_Prod_kW'].max():.0f} kW")
        results.append((ts, avg, dist))

    if show_plots:
        _plot_predict_bar(df, today_idx,
                          [(pd.Timestamp(fri_d), pred_fri)] + [(ts, v) for ts, v, _ in results],
                          weekend=True)
        all_dists = [d for _, _, d in results if d is not None]
        if all_dists:
            _plot_distribution(pd.concat(all_dists, ignore_index=True),
                               pd.Timestamp(sat_d), title="Weekend Production Distribution")

    return results


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_predict(show_plots=True):
    """
    Auto-detect day:
      Friday   → weekend model → returns [(sat,avg,dist), (sun,avg,dist), (mon,avg,dist)]
      Mon-Thu  → daily model   → returns [(tomorrow,avg,dist)]
    Each tuple: (pd.Timestamp, float kW, pd.DataFrame or None)
    """
    if datetime.date.today().weekday() == 4:
        print("  Today is Friday — using weekend model.")
        return _run_predict_weekend(show_plots)
    else:
        return _run_predict_daily(show_plots)


# =============================================================================
# PLOT HELPERS
# =============================================================================

def _plot_predict_bar(df, today_idx, forecast_list, weekend=False):
    n_bars    = 8
    hist_df   = df.iloc[max(0, today_idx - n_bars + 1): today_idx + 1]
    bar_dates = [d.date() for d in hist_df['Date']] + [ts.date() for ts, _ in forecast_list]
    bar_vals  = list(hist_df['prod_total'].values) + [v for _, v in forecast_list]
    colors    = ['steelblue'] * len(hist_df) + \
                (['gold'] + ['orange'] * (len(forecast_list)-1) if weekend else ['orange'])

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(bar_dates)), bar_vals, color=colors, edgecolor='black', alpha=0.85)
    ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8)
    ax.set_xticks(range(len(bar_dates)))
    ax.set_xticklabels([str(d) for d in bar_dates], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Daily Average Production (kW)')
    ax.set_title('Production Forecast')
    from matplotlib.patches import Patch
    handles = [Patch(color='steelblue', label='Historical'), Patch(color='orange', label='Predicted')]
    if weekend:
        handles.insert(1, Patch(color='gold', label='Friday (ref)'))
    ax.legend(handles=handles)
    plt.tight_layout(); plt.show()


def _plot_distribution(dist_df, first_date, title="Production Distribution Forecast"):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dist_df['DateTime'], dist_df['Predicted_Prod_kW'],
            color='steelblue', linewidth=1.2)
    ax.axhline(MAX_PRODUCTION_KW, color='red', linestyle='--', linewidth=0.8, label=f'Max {MAX_PRODUCTION_KW} kW')
    ax.set_xlabel('Time'); ax.set_ylabel('Predicted Production (kW)')
    ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.show()


def _plot_training(history, model, scaler_y, X_sc, y, dates, title_prefix=""):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Train'); ax.plot(history.history['val_loss'], label='Val')
    ax.set_title(f'{title_prefix} — Training Loss'); ax.legend(); plt.tight_layout(); plt.show()

    y_pred_sc = model.predict(X_sc).ravel()
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    mae  = np.mean(np.abs(y_pred - y))
    rmse = np.sqrt(np.mean((y_pred - y)**2))
    mape = np.mean(np.abs((y_pred - y) / (y + 1e-6))) * 100
    print(f"\n--- KPIs ---  RMSE={rmse:,.0f} kW  MAE={mae:,.0f} kW  MAPE={mape:.1f}%")

    dates_dt = [d.date() if hasattr(d, 'date') else d for d in dates]
    fig, ax  = plt.subplots(figsize=(14, 5))
    ax.plot(dates_dt, y,      label='Actual',    color='steelblue', lw=1)
    ax.plot(dates_dt, y_pred, label='Predicted', color='orange',    lw=1, ls='--')
    ax.set_title(f'{title_prefix} — Actual vs Predicted'); ax.legend(); plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_pred, alpha=0.5, s=15, color='steelblue')
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1, label='Perfect'); ax.legend()
    ax.set_title('Scatter'); plt.tight_layout(); plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Hydro Production Forecast')
    parser.add_argument('--train',         action='store_true', help='Train daily model')
    parser.add_argument('--train-weekend', action='store_true', help='Train weekend 4-day model')
    parser.add_argument('--train-shape',   action='store_true', help='Train intraday shape model')
    parser.add_argument('--predict',       action='store_true', help='Predict (auto weekday/Friday)')
    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.train_weekend:
        run_train_weekend()
    elif args.train_shape:
        run_train_shape()
    elif args.predict:
        run_predict()
    else:
        parser.print_help()
