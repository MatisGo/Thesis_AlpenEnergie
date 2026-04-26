"""
Plot_PV_Comparison.py
=====================
Side-by-side evaluation of the two PV estimation approaches:

  Left  — Static correction table (Lorenz 2011 / Bacher 2009)
           PV_Est = GHI × ratio[month, 15min_slot]

  Right — Deep MLP (Train_PV_NN.py)
           PV_Est = MLP(GHI, Cloud, Temp, Rain, Month_sin/cos, Hour_sin/cos, PV_t1, PV_t4)

Both methods are evaluated on the same held-out test set (last 10% of the data
by chronological order), matching the split used in Train_PV_NN.py.

Requires:
    PV_merged_2023_2025.xlsx   — enriched by Train_PV_NN.py (has weather columns)
    PV_Correction_Table.npz    — built by Build_PV_Correction_Table.py
    PV_NN_Model.keras          — trained by Train_PV_NN.py
    PV_NN_Scaler.pkl           — saved by Train_PV_NN.py

Output:
    PV_Method_Comparison.png   — 2-panel figure (same style as Train_PV_NN.py)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR  = Path(__file__).parent
RESSOURCE   = SCRIPT_DIR / '..' / 'Ressource'

PV_XLS        = RESSOURCE / 'PV_merged_2023_2025.xlsx'
TABLE_NPZ     = SCRIPT_DIR / 'PV_Correction_Table.npz'
NN_MODEL_PATH = SCRIPT_DIR / 'PV_NN_Model.keras'
NN_SCALER_PATH= SCRIPT_DIR / 'PV_NN_Scaler.pkl'
OUT_PLOT      = SCRIPT_DIR / 'PV_Method_Comparison.png'

IRR_THRESHOLD = 10.0   # W/m² — daytime filter for scatter (same as training)

# Feature order must match Train_PV_NN.py FEATURE_COLS exactly
FEATURE_COLS = [
    'GHI_Wm2', 'Cloud_Cover', 'Temperature_C', 'Precip_mm',
    'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos',
    'GHI_effective', 'GHI_sq', 'GHI_x_hour_sin', 'GHI_x_hour_cos',
    'Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn',
]

# =============================================================================
# LOAD DATA
# =============================================================================

def load_pv_data():
    """Load enriched PV xlsx (produced by Train_PV_NN.py)."""
    print(f"Loading: {PV_XLS.name} …")
    df = pd.read_excel(PV_XLS, header=0)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = next((c for c in df.columns if c.lower() in ('date', 'datum')), None)
    time_col = next((c for c in df.columns if c.lower() in ('time', 'zeit')), None)

    date_part = pd.to_datetime(df[date_col], format='%d.%m.%Y', errors='coerce')
    if date_part.isna().mean() > 0.5:
        date_part = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()

    try:
        time_part = pd.to_timedelta(df[time_col].astype(str) + ':00', errors='coerce')
        if time_part.isna().mean() > 0.5:
            time_part = pd.to_timedelta(df[time_col].astype(str), errors='coerce')
    except Exception:
        time_part = pd.to_timedelta(df[time_col].astype(str), errors='coerce')

    df['DateTime'] = date_part + time_part
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    # PV column
    pv_col = next((c for c in df.columns if 'PV' in c and 'kW' in c.lower().replace('kw','kW')), None)
    if pv_col is None:
        pv_col = next((c for c in df.columns if 'pv' in c.lower()), None)
    df['PV_kW'] = pd.to_numeric(df[pv_col], errors='coerce')

    # Weather columns (added by Train_PV_NN.py)
    for col in ['GHI_Wm2', 'Cloud_Cover', 'Temperature_C', 'Precip_mm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            print(f"  WARNING: '{col}' not found — set to 0. Re-run Train_PV_NN.py first.")
            df[col] = 0.0

    df = df.dropna(subset=['PV_kW'])
    print(f"  {len(df):,} rows | PV range: [{df['PV_kW'].min():.1f}, {df['PV_kW'].max():.1f}] kW")
    return df


# =============================================================================
# METHOD 1 — CORRECTION TABLE
# =============================================================================

def apply_correction_table(df):
    """Apply Lorenz (2011) correction table and return PV_Est array."""
    data = np.load(TABLE_NPZ)
    ratio_table = data['ratio_table'].astype(np.float32)   # (12, 96)
    print(f"  Correction table loaded: max_ratio={ratio_table.max():.4f}")

    months = df['DateTime'].dt.month.values - 1             # 0-indexed
    slots  = (df['DateTime'].dt.hour.values * 4 +
              df['DateTime'].dt.minute.values // 15)        # 0-95
    slots  = np.clip(slots, 0, 95)

    pv_est = (df['GHI_Wm2'].values * ratio_table[months, slots]).astype(np.float32)
    return pv_est


# =============================================================================
# METHOD 2 — PV DEEP NN
# =============================================================================

def apply_pv_nn(df_full, test_start_idx):
    """
    Run weather-only PV MLP inference (no AR features).

    AR lags removed from PV NN to eliminate exposure bias when the model
    is called inside CNN-LSTM (where true past PV is unavailable for
    the forecast horizon). Single deterministic pass — same computation
    at training and inference time.

    Returns only the test-set predictions (array of length n_test).
    """
    pv_model = load_model(str(NN_MODEL_PATH))
    with open(NN_SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

    df = df_full.copy().reset_index(drop=True)
    month = df['DateTime'].dt.month.values
    hour  = df['DateTime'].dt.hour.values + df['DateTime'].dt.minute.values / 60.0

    ghi      = df['GHI_Wm2'].values.astype(np.float32)
    cc       = df['Cloud_Cover'].values.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    # Season one-hot flags — must match build_features() in Train_PV_NN.py
    season_winter = np.isin(month, [12, 1, 2]).astype(np.float32)
    season_spring = np.isin(month, [3, 4, 5]).astype(np.float32)
    season_summer = np.isin(month, [6, 7, 8]).astype(np.float32)
    season_autumn = np.isin(month, [9, 10, 11]).astype(np.float32)

    feat_mat = np.column_stack([
        ghi,
        cc,
        df['Temperature_C'].values.astype(np.float32),
        df['Precip_mm'].values.astype(np.float32),
        np.sin(2 * np.pi * month / 12).astype(np.float32),
        np.cos(2 * np.pi * month / 12).astype(np.float32),
        hour_sin,
        hour_cos,
        (ghi * (1.0 - cc / 100.0)),           # GHI_effective
        ((ghi / 1000.0) ** 2),                  # GHI_sq
        (ghi * hour_sin / 1000.0),             # GHI_x_hour_sin
        (ghi * hour_cos / 1000.0),             # GHI_x_hour_cos
        season_winter,                          # Season_Winter
        season_spring,                          # Season_Spring
        season_summer,                          # Season_Summer
        season_autumn,                          # Season_Autumn
    ])

    feat_mat_s = scaler_X.transform(feat_mat)
    pv_s       = pv_model.predict(feat_mat_s, batch_size=4096, verbose=0)
    pv_all     = scaler_y.inverse_transform(pv_s).flatten().clip(min=0.0)

    # Nighttime shortcut — model trained on daytime only; force zero for GHI < threshold
    night_mask = df_full['GHI_Wm2'].values < IRR_THRESHOLD
    pv_all[night_mask] = 0.0

    return pv_all[test_start_idx:].astype(np.float32)


# =============================================================================
# METRICS
# =============================================================================

def metrics(y_true, y_pred, mask):
    mae  = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    r2   = r2_score(y_true[mask], y_pred[mask])
    return mae, rmse, r2


# =============================================================================
# PLOT
# =============================================================================

def plot_comparison(y_true, pv_table, pv_nn, day_mask):
    mae_t,  rmse_t,  r2_t  = metrics(y_true, pv_table, day_mask)
    mae_nn, rmse_nn, r2_nn = metrics(y_true, pv_nn,    day_mask)

    print("\n" + "="*60)
    print("  DAYTIME TEST SET COMPARISON")
    print("="*60)
    print(f"  Correction Table — MAE: {mae_t:.1f} kW | RMSE: {rmse_t:.1f} kW | R²: {r2_t:.3f}")
    print(f"  Deep NN (MLP)    — MAE: {mae_nn:.1f} kW | RMSE: {rmse_nn:.1f} kW | R²: {r2_nn:.3f}")
    print("="*60)

    lim = max(y_true[day_mask].max(), pv_table[day_mask].max(), pv_nn[day_mask].max()) * 1.02

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('PV Estimation — Method Comparison (Test Set, Daytime Samples)',
                 fontsize=13, fontweight='bold')

    # ── Left: Correction Table ────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_true[day_mask], pv_table[day_mask],
               alpha=0.15, s=3, color='darkorange', label='Daytime samples')
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Measured PV [kW]')
    ax.set_ylabel('Estimated PV [kW]')
    ax.set_title(
        f'Static Correction Table\n'
        f'(Lorenz 2011: GHI × ratio[month, slot])\n'
        f'MAE={mae_t:.1f} kW | RMSE={rmse_t:.1f} kW | R²={r2_t:.3f}',
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: Deep NN ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(y_true[day_mask], pv_nn[day_mask],
               alpha=0.15, s=3, color='royalblue', label='Daytime samples')
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Measured PV [kW]')
    ax.set_ylabel('Estimated PV [kW]')
    ax.set_title(
        f'Deep MLP  (MLP 128→64→32→1)\n'
        f'Features: GHI, Cloud, T, Rain, Time, PV_AR\n'
        f'MAE={mae_nn:.1f} kW | RMSE={rmse_nn:.1f} kW | R²={r2_nn:.3f}',
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_PLOT), dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\n  Plot saved to: {OUT_PLOT}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PV ESTIMATION — METHOD COMPARISON")
    print("Correction Table (Lorenz 2011) vs Deep MLP")
    print("=" * 60)

    # 1. Load full data (needed for continuous lag computation)
    print("\n--- Loading PV + weather data ---")
    df = load_pv_data()

    # Chronological 80/10/10 split — must match Train_PV_NN.py exactly
    n            = len(df)
    n_test       = max(int(n * 0.10), 1)
    n_val        = max(int(n * 0.10), 1)
    n_train      = n - n_val - n_test
    test_start   = n_train + n_val        # first index of test set in full df
    df_test      = df.iloc[test_start:].reset_index(drop=True)
    print(f"  Test set: {len(df_test):,} rows  "
          f"({df_test['DateTime'].min().date()} to {df_test['DateTime'].max().date()})")

    y_true   = df_test['PV_kW'].values.astype(np.float32)
    day_mask = df_test['GHI_Wm2'].values > IRR_THRESHOLD

    # 2. Apply correction table (test set only — no lag dependency)
    print("\n--- Method 1: Correction Table ---")
    pv_table = apply_correction_table(df_test)

    # 3. Apply PV NN on FULL df, return test slice
    #    (lags must be continuous across val/test boundary)
    print("\n--- Method 2: Deep NN ---")
    pv_nn = apply_pv_nn(df, test_start_idx=test_start)

    # 4. Plot
    print("\n--- Generating comparison plot ---")
    plot_comparison(y_true, pv_table, pv_nn, day_mask)
