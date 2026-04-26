"""
Train_PV_NN.py — Deep Neural Network for PV Production Estimation
=================================================================
Trains an MLP (Multi-Layer Perceptron) to predict local PV production (kW)
from weather inputs, replacing the static Lorenz (2011) correction table.

Why MLP over CNN for tabular weather data:
------------------------------------------
Goodfellow et al. (2016) show that CNNs exploit spatial/temporal locality
via shared-weight convolutions, which is ideal for time-series or image
inputs. For tabular weather inputs (one row = one timestamp), there is no
spatial structure to exploit. Bzdok et al. (2018) and Grinsztajn et al.
(2022, NeurIPS) demonstrate that MLPs match or outperform CNNs on tabular
regression tasks. For this PV estimation problem (10 scalar features → 1
scalar output) an MLP is therefore the scientifically appropriate choice.

Autoregressive features PV(t-1) and PV(t-4) are included because past PV
production provides information about cloud ramp events and transient
shading not fully captured by GHI alone (Lorenz et al. 2011).

Scientific references:
    Goodfellow et al. (2016) "Deep Learning", MIT Press. §9 (CNNs).
    Lorenz et al. (2011) DOI: 10.1002/pip.1033
    Bzdok et al. (2018) DOI: 10.1038/s41592-018-0019-x
    Grinsztajn et al. (2022) NeurIPS, arXiv:2207.08815

Architecture (residual MLP):
    Input (12 features)
      → project → ResBlock(256) → ResBlock(256) → ResBlock(128)
      → Dense(64, ELU) → Dense(32, ELU) → Dense(1, linear)

    Each ResBlock: Dense(N,ELU) → BN → Dropout → Dense(N,ELU) → BN + skip

Features (8 base + 4 physics interactions):
    GHI_Wm2        — Global Horizontal Irradiance (W/m²)  [primary driver]
    Cloud_Cover    — Total cloud cover (%)                 [modulates GHI]
    Temperature_C  — Air temperature 2m (°C)               [temperature derating]
    Precip_mm      — Precipitation (mm)                    [soiling / cloud proxy]
    Month_sin      — sin(2π × month/12)                    [seasonal angle]
    Month_cos      — cos(2π × month/12)
    Hour_sin       — sin(2π × hour/24)                     [diurnal angle]
    Hour_cos       — cos(2π × hour/24)
    GHI_effective  — GHI × (1 - Cloud_Cover/100)          [cloud-attenuated irradiance]
    GHI_sq         — GHI²  (normalised)                    [nonlinear panel response]
    GHI_x_hour_sin — GHI × Hour_sin                       [sun-angle × irradiance]
    GHI_x_hour_cos — GHI × Hour_cos                       [sun-angle × irradiance]

Data:
    PV measurements : PV_merged_2023_2025.xlsx  (15-min, 2023-2026)
    Weather         : fetched from Open-Meteo historical archive API
                      (ERA5 reanalysis, hourly → interpolated to 15-min)

Outputs (written to PV_Correction/):
    PV_NN_Model.keras   — trained Keras model
    PV_NN_Scaler.pkl    — fitted MinMaxScaler for features + target
    PV_NN_Training.png  — loss curves + validation scatter plot

Usage:
    python PV_Correction/Train_PV_NN.py
"""

import os
import time
import pickle
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Activation
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent          # .../PV_Correction/
RESSOURCE  = SCRIPT_DIR / '..' / 'Ressource'

# Input data
PV_XLS        = RESSOURCE / 'PV_merged_2023_2025.xlsx'
FORECAST_XLS  = RESSOURCE / 'Imported_Forecast.xlsx'   # same GHI source as CNN-LSTM pipeline

# Outputs
MODEL_PATH  = SCRIPT_DIR / 'PV_NN_Model.keras'
SCALER_PATH = SCRIPT_DIR / 'PV_NN_Scaler.pkl'
PLOT_PATH   = SCRIPT_DIR / 'PV_NN_Training.png'

# Meiringen station coordinates (Open-Meteo)
LAT = 46.7300
LON =  8.1843

# Data period (must cover PV xlsx range)
START_DATE = '2023-01-01'
END_DATE   = '2025-12-31'

# Irradiance threshold: rows with GHI below this are EXCLUDED from training.
# Nighttime samples (GHI=0, PV=0) dominate the dataset numerically and pull
# all predictions toward zero, causing systematic underestimation of peak PV.
# At inference, PV_Est is set to 0 directly for GHI < threshold (physically
# correct; no model call needed).
IRR_TRAIN_THRESHOLD  = 10.0   # W/m² — exclude night from training
IRR_REPORT_THRESHOLD = 10.0   # W/m² — same value used for scatter plot filtering

# Training
EPOCHS        = 200
BATCH_SIZE    = 512
LEARNING_RATE = 3e-4
PATIENCE      = 20

# Sample weight applied to daytime rows during training.
# The model trains on ALL samples so nighttime inputs (GHI=0) teach
# the network that zero irradiance → zero PV.  Daytime rows are
# up-weighted so peak-production errors dominate the loss.
DAYTIME_WEIGHT = 5.0

# L2 regularisation on Dense layer kernels — controls overfitting
# without reducing model depth.
L2_REG = 1e-4
VAL_SPLIT     = 0.15

# Feature columns (must match inference in CNN_LSTM_Prediction_TEST.py)
# AR features PV_t1/PV_t4 deliberately excluded: they cause exposure bias
# (trained on true PV lags, but CNN-LSTM calls the model with autoregressive
# estimates → systematic mismatch). A weather-only model is deterministic
# at inference and avoids this problem entirely.
FEATURE_COLS = [
    # Base weather + time
    'GHI_Wm2', 'Cloud_Cover', 'Temperature_C', 'Precip_mm',
    'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos',
    # Physics-informed interaction features
    'GHI_effective',   # GHI × (1 - Cloud_Cover/100) — cloud attenuation
    'GHI_sq',          # GHI² normalised — nonlinear panel response
    'GHI_x_hour_sin',  # GHI × Hour_sin — sun angle × irradiance
    'GHI_x_hour_cos',  # GHI × Hour_cos — sun angle × irradiance
    # Meteorological season flags (one-hot, 4 seasons)
    # Provide discrete seasonal context that Month_sin/cos cannot fully encode
    # (e.g. solar panel performance differs between Spring and Autumn even at
    # the same monthly angle due to snow cover, soiling, and temperature).
    'Season_Winter',   # Dec, Jan, Feb
    'Season_Spring',   # Mar, Apr, May
    'Season_Summer',   # Jun, Jul, Aug
    'Season_Autumn',   # Sep, Oct, Nov
]
TARGET_COL = 'PV_kW'


# =============================================================================
# STEP 1 — FETCH WEATHER FROM OPEN-METEO HISTORICAL ARCHIVE API
# =============================================================================

def fetch_openmeteo_weather(lat: float, lon: float,
                            start: str, end: str) -> pd.DataFrame:
    """
    Download hourly ERA5 reanalysis weather from the Open-Meteo archive API.

    Variables requested:
        shortwave_radiation   — GHI (W/m²)
        cloud_cover           — Total cloud cover (%)
        temperature_2m        — Air temperature at 2m (°C)
        precipitation         — Hourly precipitation (mm)

    Returns
    -------
    df_15 : pd.DataFrame
        15-min regular grid (linear interpolation from hourly),
        columns: DateTime, GHI_Wm2, Cloud_Cover, Temperature_C, Precip_mm
    """
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude':  lat,
        'longitude': lon,
        'start_date': start,
        'end_date':   end,
        'hourly': ','.join([
            'shortwave_radiation',
            'cloud_cover',
            'temperature_2m',
            'precipitation',
        ]),
        'timezone': 'Europe/Zurich',
    }

    print(f"  Fetching Open-Meteo ERA5 data for ({lat}, {lon}) "
          f"from {start} to {end} …")
    t0 = time.time()

    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Open-Meteo API request failed: {e}")

    data = resp.json()
    hourly = data.get('hourly', {})
    if not hourly or 'time' not in hourly:
        raise RuntimeError("Open-Meteo response missing 'hourly' data.")

    df_h = pd.DataFrame({
        'DateTime':     pd.to_datetime(hourly['time']),
        'GHI_Wm2':      hourly.get('shortwave_radiation', []),
        'Cloud_Cover':  hourly.get('cloud_cover', []),
        'Temperature_C': hourly.get('temperature_2m', []),
        'Precip_mm':    hourly.get('precipitation', []),
    })

    print(f"  Fetched {len(df_h):,} hourly rows in {time.time()-t0:.1f}s")
    print(f"  Date range: {df_h['DateTime'].min()} → {df_h['DateTime'].max()}")

    # ── Interpolate hourly → 15-min ──────────────────────────────────────────
    full_15min = pd.date_range(
        start=df_h['DateTime'].min(),
        end=df_h['DateTime'].max(),
        freq='15min',
    )
    df_h = df_h.set_index('DateTime')
    df_15 = df_h.reindex(df_h.index.union(full_15min)).interpolate(method='time')
    df_15 = df_15.reindex(full_15min)
    df_15.index.name = 'DateTime'
    df_15 = df_15.reset_index()

    # Clip physical bounds after interpolation
    df_15['GHI_Wm2']    = df_15['GHI_Wm2'].clip(lower=0.0)
    df_15['Cloud_Cover'] = df_15['Cloud_Cover'].clip(0.0, 100.0)
    df_15['Precip_mm']  = df_15['Precip_mm'].clip(lower=0.0)

    print(f"  Interpolated to {len(df_15):,} rows at 15-min resolution")
    return df_15


# =============================================================================
# STEP 1b — LOAD WEATHER FROM IMPORTED_FORECAST.XLSX (preferred source)
# =============================================================================

def load_imported_forecast_weather(path: Path) -> pd.DataFrame:
    """
    Load weather from Imported_Forecast.xlsx — the SAME GHI source used by
    CNN_LSTM_Prediction_TEST.py at training and inference time.

    Training the PV NN on this source eliminates the ERA5 vs forecast-model
    GHI mismatch that caused degradation of the net load forecast.

    Resamples from 5-min → 15-min (mean for intensive quantities, sum for precip).

    Returns DataFrame with columns:
        DateTime, GHI_Wm2, Cloud_Cover, Temperature_C, Precip_mm
    """
    print(f"  Loading weather from: {path.name} …")
    df_w = pd.read_excel(path, sheet_name='Weather_Data')
    df_w['DateTime'] = pd.to_datetime(df_w['Time'], format='%d.%m.%Y %H:%M:%S',
                                      errors='coerce')
    df_w = df_w.dropna(subset=['DateTime']).sort_values('DateTime')

    # Rename to standard internal column names
    rename_map = {
        'Irradiance_Wm2': 'GHI_Wm2',
        'Cloud_Cover_Pct': 'Cloud_Cover',
        'Temperature_C':   'Temperature_C',
        'Rain_Sum_mm':     'Precip_mm',
    }
    df_w = df_w.rename(columns=rename_map)

    needed = ['GHI_Wm2', 'Cloud_Cover', 'Temperature_C', 'Precip_mm']
    missing = [c for c in needed if c not in df_w.columns]
    if missing:
        raise ValueError(f"Imported_Forecast.xlsx missing columns: {missing}. "
                         f"Re-run get_weather_data.py with cloud cover enabled.")

    df_w = df_w.set_index('DateTime')[needed]

    # Resample 5-min → 15-min
    df_15 = df_w.resample('15min').agg({
        'GHI_Wm2':       'mean',
        'Cloud_Cover':   'mean',
        'Temperature_C': 'mean',
        'Precip_mm':     'sum',    # precipitation is additive over the interval
    })
    df_15['GHI_Wm2']    = df_15['GHI_Wm2'].clip(lower=0.0)
    df_15['Cloud_Cover'] = df_15['Cloud_Cover'].clip(0.0, 100.0)
    df_15['Precip_mm']  = df_15['Precip_mm'].clip(lower=0.0)

    df_15 = df_15.reset_index()
    coverage_days = (df_15['DateTime'].max() - df_15['DateTime'].min()).days
    print(f"  {len(df_15):,} rows | coverage: {coverage_days} days "
          f"({df_15['DateTime'].min().date()} to {df_15['DateTime'].max().date()})")
    return df_15


def blend_weather(df_primary: pd.DataFrame, df_fallback: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two weather DataFrames: primary (Imported_Forecast) takes priority;
    ERA5 fallback fills dates not covered by primary.

    Both must have columns: DateTime, GHI_Wm2, Cloud_Cover, Temperature_C, Precip_mm
    """
    cols = ['GHI_Wm2', 'Cloud_Cover', 'Temperature_C', 'Precip_mm']
    df_p = df_primary.set_index('DateTime')[cols]
    df_f = df_fallback.set_index('DateTime')[cols]

    # Combine: primary overrides fallback where both exist
    combined = df_f.combine_first(df_p)   # fallback first, then primary overwrites
    combined = df_p.combine_first(combined)  # ensure primary wins
    combined = combined.sort_index().reset_index()
    print(f"  Blended weather: {len(combined):,} rows total "
          f"(primary: {len(df_p):,}, fallback filled gaps)")
    return combined


# =============================================================================
# STEP 2 — LOAD PV DATA FROM XLSX
# =============================================================================

def load_pv_xlsx(path: Path) -> pd.DataFrame:
    """
    Load PV_merged_2023_2025.xlsx.

    Expected columns:  Date (dd.mm.yyyy), Time (HH:MM), PV Production [kW]

    Returns
    -------
    df_pv : pd.DataFrame
        Columns: DateTime, PV_kW
    """
    print(f"  Loading PV data from: {path.name} …")
    df = pd.read_excel(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"  Columns found: {list(df.columns)}")
    print(f"  Rows: {len(df):,}")

    # ── Parse DateTime ───────────────────────────────────────────────────────
    # Handle both possible column names (Date/Datum, Time/Zeit)
    date_col = next((c for c in df.columns
                     if c.lower() in ('date', 'datum')), None)
    time_col = next((c for c in df.columns
                     if c.lower() in ('time', 'zeit')), None)
    pv_col   = next((c for c in df.columns
                     if 'PV' in c and 'kW' in c.lower().replace('kw', 'kW')), None)

    # Fallback: find PV column by 'pv' keyword
    if pv_col is None:
        pv_col = next((c for c in df.columns if 'pv' in c.lower()), None)

    if date_col is None or time_col is None or pv_col is None:
        raise ValueError(
            f"Cannot identify Date/Time/PV columns.\n"
            f"Available: {list(df.columns)}"
        )

    print(f"  Using: date='{date_col}', time='{time_col}', PV='{pv_col}'")

    # Parse date: dd.mm.yyyy or Excel serial
    try:
        date_part = pd.to_datetime(df[date_col], format='%d.%m.%Y', errors='coerce')
        if date_part.isna().mean() > 0.5:
            date_part = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    except Exception:
        date_part = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()

    # Parse time: HH:MM or timedelta
    try:
        time_part = pd.to_timedelta(df[time_col].astype(str) + ':00', errors='coerce')
        if time_part.isna().mean() > 0.5:
            time_part = pd.to_timedelta(df[time_col].astype(str), errors='coerce')
    except Exception:
        time_part = pd.to_timedelta(df[time_col].astype(str), errors='coerce')

    df['DateTime'] = date_part + time_part
    df['PV_kW']    = pd.to_numeric(df[pv_col], errors='coerce')

    df_pv = df[['DateTime', 'PV_kW']].dropna(subset=['DateTime'])
    df_pv = df_pv.sort_values('DateTime').reset_index(drop=True)

    print(f"  Rows after DateTime parse: {len(df_pv):,}")
    print(f"  PV range: [{df_pv['PV_kW'].min():.1f}, {df_pv['PV_kW'].max():.1f}] kW")
    print(f"  Missing PV values: {df_pv['PV_kW'].isna().sum()}")
    return df_pv


# =============================================================================
# STEP 3 — MERGE PV + WEATHER AND ENRICH XLSX
# =============================================================================

def merge_and_enrich(df_pv: pd.DataFrame,
                     df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PV production with weather on DateTime (left join from PV index).
    Overwrites PV xlsx with the merged data (adds weather columns).

    Returns the merged DataFrame, rows with missing PV already dropped.
    """
    print("  Merging PV and weather data …")

    # Align on 15-min boundaries (round PV timestamps to nearest 15-min)
    df_pv = df_pv.copy()
    df_pv['DateTime'] = df_pv['DateTime'].dt.round('15min')

    df_weather = df_weather.copy()
    df_weather['DateTime'] = df_weather['DateTime'].dt.round('15min')

    df = pd.merge(df_pv, df_weather, on='DateTime', how='left')

    before = len(df)
    df = df.dropna(subset=['PV_kW'])   # drop rows with missing PV production
    after  = len(df)
    print(f"  Dropped {before - after:,} rows with missing PV (kept {after:,})")

    # Forward-fill any small gaps in weather (≤2 slots)
    weather_cols = ['GHI_Wm2', 'Cloud_Cover', 'Temperature_C', 'Precip_mm']
    df[weather_cols] = df[weather_cols].fillna(method='ffill', limit=2)
    df[weather_cols] = df[weather_cols].fillna(0.0)

    # ── Overwrite PV xlsx with enriched data ─────────────────────────────────
    print(f"  Overwriting {PV_XLS.name} with weather columns …")
    export_df = df[['DateTime', 'PV_kW'] + weather_cols].copy()
    export_df.insert(0, 'Date', export_df['DateTime'].dt.strftime('%d.%m.%Y'))
    export_df.insert(1, 'Time', export_df['DateTime'].dt.strftime('%H:%M'))
    export_df = export_df.drop(columns=['DateTime'])
    export_df.to_excel(PV_XLS, index=False)
    print(f"  Saved enriched PV data to: {PV_XLS}")

    return df


# =============================================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features and physics-informed interaction terms.

    Base features:
        Month_sin/cos — seasonal cycle
        Hour_sin/cos  — diurnal cycle

    Physics interactions (Antonanzas et al. 2016, Solar Energy 136):
        GHI_effective  = GHI × (1 - Cloud_Cover/100)
            The cloud-attenuation formula; this multiplicative relationship
            is hard for a shallow MLP to discover implicitly but trivial
            once given explicitly.
        GHI_sq         = (GHI / 1000)²
            Captures the mild nonlinearity in panel I-V response at
            high irradiance; normalised by 1000² to keep scale ~0-1.
        GHI_x_hour_sin = GHI × Hour_sin
        GHI_x_hour_cos = GHI × Hour_cos
            Encode the interaction between irradiance magnitude and the
            position of the sun in the diurnal cycle (morning vs afternoon
            asymmetry due to valley shading in Meiringen).
    """
    df = df.copy().sort_values('DateTime').reset_index(drop=True)

    month = df['DateTime'].dt.month.values
    hour  = (df['DateTime'].dt.hour +
             df['DateTime'].dt.minute / 60.0).values

    # ── Cyclical time features ───────────────────────────────────────────────
    df['Month_sin'] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df['Month_cos'] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df['Hour_sin']  = hour_sin
    df['Hour_cos']  = hour_cos

    # ── Physics-informed interaction features ────────────────────────────────
    ghi = df['GHI_Wm2'].values.astype(np.float32)
    cc  = df['Cloud_Cover'].values.astype(np.float32)

    df['GHI_effective']  = (ghi * (1.0 - cc / 100.0)).astype(np.float32)
    df['GHI_sq']         = ((ghi / 1000.0) ** 2).astype(np.float32)
    df['GHI_x_hour_sin'] = (ghi * hour_sin / 1000.0).astype(np.float32)
    df['GHI_x_hour_cos'] = (ghi * hour_cos / 1000.0).astype(np.float32)

    # ── Meteorological season one-hot flags ──────────────────────────────────
    # Discrete season flags give the model an explicit categorical split that
    # Month_sin/cos cannot represent: a sine/cosine at month=3 (March) and
    # month=9 (September) have opposite signs but similar |value|, yet PV
    # production differs significantly (snow residue, panel soiling, daylight
    # hours).  One-hot encoding lets the network learn fully independent
    # season-specific offsets.
    #   Winter : Dec (12), Jan (1), Feb (2)
    #   Spring : Mar (3), Apr (4), May (5)
    #   Summer : Jun (6), Jul (7), Aug (8)
    #   Autumn : Sep (9), Oct (10), Nov (11)
    m = df['DateTime'].dt.month.values
    df['Season_Winter'] = np.isin(m, [12, 1, 2]).astype(np.float32)
    df['Season_Spring'] = np.isin(m, [3, 4, 5]).astype(np.float32)
    df['Season_Summer'] = np.isin(m, [6, 7, 8]).astype(np.float32)
    df['Season_Autumn'] = np.isin(m, [9, 10, 11]).astype(np.float32)

    return df


# =============================================================================
# STEP 5 — BUILD MLP MODEL
# =============================================================================

def _res_block(x, units: int, dropout: float, reg: float, name: str):
    """
    Residual block: Dense(L2) → BN → ELU → Dropout → Dense(L2) → BN + skip.

    ELU chosen over ReLU: avoids dying-neuron problem and has smoother
    gradient (Clevert et al. 2016 ICLR arXiv:1511.07289).
    Skip connection stabilises deep-network training by allowing gradients
    to flow directly through identity shortcuts (He et al. 2016 CVPR).
    L2 kernel regularisation adds a weight-decay penalty that prevents
    individual neurons from memorising training samples.
    """
    shortcut = x
    x = Dense(units, kernel_regularizer=l2(reg), name=f'{name}_fc1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('elu', name=f'{name}_elu1')(x)
    x = Dropout(dropout, name=f'{name}_drop')(x)
    x = Dense(units, kernel_regularizer=l2(reg), name=f'{name}_fc2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('elu', name=f'{name}_elu2')(x)
    return x


def build_mlp(n_features: int) -> tf.keras.Model:
    """
    Compact regularised residual MLP for PV regression.

    Sizing rationale
    ----------------
    Previous 512-unit, 5-block model had ~1.5 M parameters for ~50 k training
    samples → 30:1 parameter-to-sample ratio → severe overfitting (val loss
    diverged from epoch 5).  Rule of thumb for tabular regression: keep the
    ratio below 1:1, so with 50 k samples the model should have ≤50 k params.

    Architecture (~120 k parameters):
        Input (16 features)
          → Dense(256, ELU) + BN + L2     — project to residual dimension
          → ResBlock(256, drop=0.35, L2)  — block 1
          → ResBlock(256, drop=0.35, L2)  — block 2
          → Dense(128, ELU) + L2          — narrow
          → Dense(64,  ELU) + L2
          → Dense(1, softplus)            — always ≥ 0 (physics constraint)

    Training strategy
    -----------------
    Trained on ALL samples (day + night).  Nighttime inputs have
    GHI = GHI_effective = GHI_sq = GHI_x_hour = 0, teaching the network
    boundary condition: zero irradiance → zero PV.
    Daytime rows receive DAYTIME_WEIGHT = 5× so peak-production accuracy
    dominates the loss — without discarding the night boundary information.
    The scaler is fitted on ALL training data, so y_min = 0 and
    softplus(~0) inverse-transforms cleanly to ~0 kW.

    References:
        He et al. (2016) CVPR arXiv:1512.03385
        Clevert et al. (2016) ICLR arXiv:1511.07289
        Grinsztajn et al. (2022) NeurIPS arXiv:2207.08815
    """
    inp = Input(shape=(n_features,), name='weather_input')

    # ── Projection ───────────────────────────────────────────────────────────
    x = Dense(256, kernel_regularizer=l2(L2_REG), name='proj')(inp)
    x = BatchNormalization(name='proj_bn')(x)
    x = Activation('elu', name='proj_elu')(x)

    # ── 2 residual blocks at 256 units ───────────────────────────────────────
    x = _res_block(x, 256, dropout=0.35, reg=L2_REG, name='res1')
    x = _res_block(x, 256, dropout=0.35, reg=L2_REG, name='res2')

    # ── Narrowing head ───────────────────────────────────────────────────────
    x = Dense(128, activation='elu', kernel_regularizer=l2(L2_REG), name='head_fc1')(x)
    x = Dense(64,  activation='elu', kernel_regularizer=l2(L2_REG), name='head_fc2')(x)

    # softplus = log(1 + exp(x)) — smooth, always ≥ 0, gradient everywhere.
    # PV production cannot be negative; this enforces the physical constraint
    # at the model level instead of a post-hoc clip that kills gradients.
    out = Dense(1, activation='softplus', name='pv_output')(x)

    model = Model(inputs=inp, outputs=out, name='PV_ResMLP')
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='huber',
        metrics=['mae'],
    )
    model.summary()
    return model


# =============================================================================
# STEP 6 — TRAIN AND EVALUATE
# =============================================================================

def train_model(df: pd.DataFrame):
    """
    Scale features, split train/val/test (chronological), train MLP, save outputs.

    Training strategy — ALL samples, weighted loss
    -----------------------------------------------
    Previous approach filtered training to daytime only, which caused three
    compounding problems:
      (a) ~50 % of data discarded → severe overfitting with a 512-unit model
      (b) Scaler fitted on daytime-only data → nighttime inputs fall outside
          the [0,1] normalised range → out-of-distribution inference
      (c) Softplus minimum output inverse-transformed to scaler data_min,
          creating a non-zero floor for the lowest predictions

    Instead we train on ALL samples and use sample_weight to tell the model
    that daytime errors matter DAYTIME_WEIGHT× more than nighttime errors:
      - Nighttime rows (GHI ≈ 0) have PV = 0.  With GHI-based features all
        equal to 0, the network naturally learns GHI=0 → PV≈0 from these
        samples — they are not noise, they are boundary-condition data.
      - Scaler fitted on all training data guarantees y_min = 0, so
        softplus ≈ 0 correctly inverse-transforms to ~0 kW.
      - Val set is also unfiltered, giving a stable, unbiased loss signal.

    Metrics are reported for daytime test samples only (GHI > threshold),
    which is where accuracy matters for the energy-management application.
    """
    # ── Prepare arrays ───────────────────────────────────────────────────────
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    # Chronological 80/10/10 split (no leakage from future)
    n       = len(X)
    n_test  = max(int(n * 0.10), 1)
    n_val   = max(int(n * 0.10), 1)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train],              y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],        y[n_train+n_val:]

    # Daytime masks — used for sample weights and reporting, not for filtering
    ghi_idx   = FEATURE_COLS.index('GHI_Wm2')
    day_train = X_train[:, ghi_idx] > IRR_TRAIN_THRESHOLD
    day_val   = X_val[:,   ghi_idx] > IRR_TRAIN_THRESHOLD
    day_test  = X_test[:,  ghi_idx] > IRR_REPORT_THRESHOLD

    print(f"\n  Train: {len(X_train):,} total | {day_train.sum():,} daytime "
          f"({100*day_train.mean():.0f}%)")
    print(f"  Val  : {len(X_val):,} total | {day_val.sum():,} daytime "
          f"({100*day_val.mean():.0f}%)")
    print(f"  Test : {len(X_test):,} total | {day_test.sum():,} daytime "
          f"({100*day_test.mean():.0f}%)")

    # ── Scale — fitted on ALL training data ──────────────────────────────────
    # With nighttime zeros included, y_min = 0 naturally.
    # softplus output of ~0 → scaled 0 → inverse_transform → 0 kW.  Correct.
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s   = scaler_y.transform(y_val)

    # Save scalers
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    print(f"  Scalers saved to: {SCALER_PATH}")

    # ── Sample weights — daytime rows are DAYTIME_WEIGHT× more important ─────
    sample_weight = np.where(day_train, DAYTIME_WEIGHT, 1.0).astype(np.float32)
    print(f"  Sample weights: daytime={DAYTIME_WEIGHT:.0f}x, nighttime=1x")

    # ── Build & train ────────────────────────────────────────────────────────
    model = build_mlp(n_features=len(FEATURE_COLS))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    print(f"\n  Training for up to {EPOCHS} epochs …")
    t0 = time.time()
    history = model.fit(
        X_train_s, y_train_s,
        sample_weight=sample_weight,
        validation_data=(X_val_s, y_val_s),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed/60:.1f} min "
          f"({len(history.history['loss'])} epochs)")

    # ── Evaluate on test set ─────────────────────────────────────────────────
    y_pred_s = model.predict(X_test_s, verbose=0)
    y_pred   = scaler_y.inverse_transform(y_pred_s).flatten().clip(min=0.0)
    y_true   = y_test.flatten()

    # day_test was already computed above; re-use it here
    mae_all  = mean_absolute_error(y_true, y_pred)
    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_all   = r2_score(y_true, y_pred)

    if day_test.sum() > 0:
        mae_day  = mean_absolute_error(y_true[day_test], y_pred[day_test])
        rmse_day = np.sqrt(mean_squared_error(y_true[day_test], y_pred[day_test]))
        r2_day   = r2_score(y_true[day_test], y_pred[day_test])
    else:
        mae_day = rmse_day = r2_day = float('nan')

    print("\n" + "=" * 60)
    print("  TEST SET METRICS")
    print("=" * 60)
    print(f"  All samples  — MAE: {mae_all:.2f} kW | RMSE: {rmse_all:.2f} kW | R²: {r2_all:.3f}")
    print(f"  Daytime only — MAE: {mae_day:.2f} kW | RMSE: {rmse_day:.2f} kW | R²: {r2_day:.3f}")
    print("=" * 60)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_results(history, y_true, y_pred, day_test,
                  mae_day, rmse_day, r2_day)

    print(f"\n  Model saved to: {MODEL_PATH}")
    return model, scaler_X, scaler_y, history


def _plot_results(history, y_true, y_pred, day_mask,
                  mae_day, rmse_day, r2_day):
    """2-panel figure: training loss curves + scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PV Deep NN — Training Results\n'
                 'ResMLP (256→256→128→64) | Features: GHI, Cloud, T, Rain, Time, Season | '
                 'Weighted loss (day x5)',
                 fontsize=11, fontweight='bold')

    # ── Panel 1: Loss curves ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history.history['loss'],     label='Train loss', color='steelblue')
    ax.plot(history.history['val_loss'], label='Val loss',   color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Scatter (daytime only) ──────────────────────────────────────
    ax = axes[1]
    ax.scatter(y_true[day_mask], y_pred[day_mask],
               alpha=0.15, s=3, color='royalblue', label='Daytime samples')
    lim = max(y_true[day_mask].max() if day_mask.sum() > 0 else 1,
              y_pred[day_mask].max() if day_mask.sum() > 0 else 1)
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Measured PV [kW]')
    ax.set_ylabel('Predicted PV [kW]')
    ax.set_title(f'Daytime Scatter — Test set\n'
                 f'MAE={mae_day:.1f} kW | RMSE={rmse_day:.1f} kW | R²={r2_day:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PLOT_PATH), dpi=200, bbox_inches='tight')
    plt.show()
    print(f"  Training plot saved to: {PLOT_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TRAIN PV DEEP NN — MLP for PV Production Estimation")
    print("Replacing Lorenz (2011) correction table with a learned model")
    print("=" * 70)

    # ── Step 1: Load weather — use Imported_Forecast.xlsx (same GHI source as CNN-LSTM)
    # Falls back to ERA5 archive only for dates not covered by Imported_Forecast.
    # This is critical: PV NN must train on the same GHI distribution it will
    # receive at inference time inside CNN_LSTM_Prediction_TEST.py.
    print("\n--- Step 1: Load weather (Imported_Forecast.xlsx preferred) ---")
    MIN_COVERAGE_DAYS = 180   # require at least 6 months from Imported_Forecast

    if FORECAST_XLS.exists():
        try:
            df_fc = load_imported_forecast_weather(FORECAST_XLS)
            coverage = (df_fc['DateTime'].max() - df_fc['DateTime'].min()).days
            if coverage >= MIN_COVERAGE_DAYS:
                print(f"  Using Imported_Forecast.xlsx as primary source ({coverage} days).")
                # Fetch ERA5 to gap-fill periods before Imported_Forecast coverage
                fc_start = df_fc['DateTime'].min().strftime('%Y-%m-%d')
                if fc_start > START_DATE:
                    print(f"  ERA5 needed for gap before {fc_start} …")
                    df_era5 = fetch_openmeteo_weather(LAT, LON, START_DATE, fc_start)
                    df_weather = blend_weather(df_fc, df_era5)
                else:
                    df_weather = df_fc
            else:
                print(f"  Imported_Forecast only has {coverage} days — fetching full ERA5 …")
                df_weather = fetch_openmeteo_weather(LAT, LON, START_DATE, END_DATE)
        except Exception as e:
            print(f"  WARNING: Could not load Imported_Forecast.xlsx ({e}). Falling back to ERA5.")
            df_weather = fetch_openmeteo_weather(LAT, LON, START_DATE, END_DATE)
    else:
        print(f"  Imported_Forecast.xlsx not found — fetching ERA5 from Open-Meteo archive.")
        df_weather = fetch_openmeteo_weather(LAT, LON, START_DATE, END_DATE)

    # ── Step 2: Load PV measurements ─────────────────────────────────────────
    print("\n--- Step 2: Load PV measurements ---")
    df_pv = load_pv_xlsx(PV_XLS)

    # ── Step 3: Merge, clean, enrich xlsx ────────────────────────────────────
    print("\n--- Step 3: Merge PV + weather ---")
    df = merge_and_enrich(df_pv, df_weather)

    # ── Step 4: Feature engineering ──────────────────────────────────────────
    print("\n--- Step 4: Feature engineering ---")
    df = build_features(df)

    n_day = (df['GHI_Wm2'] > IRR_REPORT_THRESHOLD).sum()
    print(f"  Total samples: {len(df):,}  |  Daytime (GHI>{IRR_REPORT_THRESHOLD}): {n_day:,}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Target range: [{df[TARGET_COL].min():.1f}, {df[TARGET_COL].max():.1f}] kW")

    # ── Step 5+6: Train ───────────────────────────────────────────────────────
    print("\n--- Step 5: Build and train MLP ---")
    model, scaler_X, scaler_y, history = train_model(df)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Scaler : {SCALER_PATH}")
    print(f"  Plot   : {PLOT_PATH}")
    print("=" * 70)
    print("\nNext: update CNN_LSTM_Prediction_TEST.py to call PV_NN_Model.keras")
    print("      instead of the Lorenz correction table.")
