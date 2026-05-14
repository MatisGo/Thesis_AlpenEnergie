"""
generate_consumption_forecast_2025.py
--------------------------------------
Batch-generates the CNN-LSTM day-ahead consumption forecast for every day
in 2025 (or whatever period the data covers) and saves a 5-min CSV for
import into matis_2025_.xlsx.

Strategy (true day-ahead, no look-ahead):
  For each target date D (Jan-1 to Dec-31 2025):
    - Prediction start  : midnight of D-1  (07:00 would be more realistic, but midnight is conservative)
    - Encoder lookback  : 7 days ending at midnight of D-1
    - Model output      : 48-hour prediction  (steps 0-287 = D-1, steps 288-575 = D)
    - Exported slice    : steps 288-575  → covers target date D  (288 steps × 5 min = 24 h)

The exported value is Load_Is (net load = gross load - local PV), which is the
quantity submitted in the DA Fahrplan net position.

Output
------
  Ressource/Consumption_Forecast_2025.csv
    Columns : DateTime, Forecast_Consumption_kW
    Resolution : 5-min (288 rows per day)

KPIs are computed against the actual Load_Is for all steps where both values exist.

Usage
-----
  cd Cons_Prediction/CNN_LSTM
  python generate_consumption_forecast_2025.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Resolve paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Re-use everything from the main prediction script
from CNN_LSTM_Prediction import (
    load_pv_correction_table,
    load_data,
    build_stat_profiles,
    load_config,
    compute_metrics,
    LOOKBACK_STEPS,
    FORECAST_HORIZON,
    STEPS_PER_HOUR,
    INPUT_FEATURES,
    STAT_N_WEEKS,
    MODEL_PATH,
    CONFIG_PATH,
)

import tensorflow as tf
from tensorflow.keras.models import load_model

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_YEAR  = 2025
DAY_SKIP     = 24 * STEPS_PER_HOUR           # 288 steps — skip D-1 portion
DAY_STEPS    = FORECAST_HORIZON - DAY_SKIP   # 288 steps — keep day-D portion
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, 'Ressource', 'Consumption_Forecast_2025.csv')


# ==============================================================================
def main():
    print("=" * 65)
    print(f"  CNN-LSTM  |  Full-year consumption forecast  |  {TARGET_YEAR}")
    print("=" * 65)

    # 1. Load model and scalers
    print("\n[1] Loading model and config ...")
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        print("  Run:  python CNN_LSTM_Prediction.py --train")
        sys.exit(1)

    model = load_model(MODEL_PATH, compile=False)
    scaler_X, scaler_y, scaler_stat, scaler_pv = load_config()

    # Read STAT_N_WEEKS from the saved config (robust against retraining with different value)
    cfg_raw     = np.load(CONFIG_PATH, allow_pickle=True)
    n_weeks_cfg = int(cfg_raw['stat_n_weeks'])
    n_feat      = int(cfg_raw['scaler_X_min_'].shape[0])
    print(f"  Model STAT_N_WEEKS = {n_weeks_cfg},  n_features = {n_feat}")

    # 2. Load and preprocess data
    print("\n[2] Loading data ...")
    ratio_table = load_pv_correction_table()
    df          = load_data(ratio_table=ratio_table)

    load_values = df['Load_Is'].values.astype(np.float32)
    pv_values   = df['PV_Est'].values.astype(np.float32)
    features    = df[INPUT_FEATURES].values.astype(np.float32)
    dt_index    = pd.DatetimeIndex(df['DateTime'].values)

    # 3. Identify valid prediction windows
    print(f"\n[3] Building prediction windows for {TARGET_YEAR} ...")
    target_dates = pd.date_range(f'{TARGET_YEAR}-01-01', f'{TARGET_YEAR}-12-31', freq='D')

    valid = []   # list of (target_date, loc_in_df)
    skipped = []

    for d in target_dates:
        pred_start = d - pd.Timedelta(days=1)   # midnight of D-1

        # Find nearest index in df
        loc = dt_index.searchsorted(pred_start)
        if loc >= len(dt_index):
            skipped.append((d, 'beyond data end'))
            continue
        if abs((dt_index[loc] - pred_start).total_seconds()) > 600:
            skipped.append((d, 'no data at midnight D-1'))
            continue
        if loc < LOOKBACK_STEPS:
            skipped.append((d, 'insufficient lookback'))
            continue

        valid.append((d, int(loc)))

    print(f"  Valid   : {len(valid)} / {len(target_dates)} days")
    if skipped:
        print(f"  Skipped : {len(skipped)} days")
        for d, reason in skipped[:5]:
            print(f"    {d.date()}  —  {reason}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped)-5} more")

    if not valid:
        print("  ERROR: No valid prediction windows. Check that data covers 2024.")
        sys.exit(1)

    n = len(valid)

    # 4. Build batch input arrays
    print(f"\n[4] Building batch inputs ({n} samples) ...")

    X_batch        = np.zeros((n, LOOKBACK_STEPS, n_feat), dtype=np.float32)
    pv_batch       = np.zeros((n, FORECAST_HORIZON),       dtype=np.float32)
    start_idx_arr  = np.zeros(n,                           dtype=np.int64)

    for i, (d, loc) in enumerate(valid):
        # Encoder: 7-day lookback ending at loc
        x_slice = features[loc - LOOKBACK_STEPS : loc].copy()
        if np.isnan(x_slice).any():
            x_slice = (pd.DataFrame(x_slice, columns=INPUT_FEATURES)
                       .interpolate().bfill().ffill().values.astype(np.float32))
        X_batch[i]        = x_slice
        start_idx_arr[i]  = loc

        # PV estimate for forecast horizon starting at loc
        end_pv   = min(loc + FORECAST_HORIZON, len(pv_values))
        pv_slice = pv_values[loc : end_pv]
        if len(pv_slice) < FORECAST_HORIZON:
            pv_slice = np.pad(pv_slice, (0, FORECAST_HORIZON - len(pv_slice)))
        pv_batch[i] = pv_slice

    # 5. Statistical profiles (computed in one call for all samples)
    print("[5] Building statistical profiles ...")
    stat_batch = build_stat_profiles(
        load_values, start_idx_arr,
        horizon=FORECAST_HORIZON,
        n_weeks=n_weeks_cfg,
    )   # shape: (n, FORECAST_HORIZON, n_weeks_cfg)

    # 6. Scale inputs
    print("[6] Scaling inputs ...")

    # Scale X_batch: apply MinMaxScaler transform via broadcasting
    X_scaled = X_batch * scaler_X.scale_ + scaler_X.min_    # same op as scaler.transform

    # Scale stat_batch
    stat_scaled = scaler_stat.transform(
        stat_batch.reshape(-1, n_weeks_cfg)
    ).reshape(n, FORECAST_HORIZON, n_weeks_cfg).astype(np.float32)

    # Scale pv_batch
    pv_scaled = scaler_pv.transform(
        pv_batch.reshape(-1, 1)
    ).reshape(n, FORECAST_HORIZON, 1).astype(np.float32)

    # 7. Batch inference
    print(f"[7] Running model.predict on {n} samples ...")
    y_pred_scaled = model.predict(
        [X_scaled, stat_scaled, pv_scaled],
        batch_size=32,
        verbose=1,
    )
    y_pred = scaler_y.inverse_transform(y_pred_scaled)   # (n, FORECAST_HORIZON)

    # 8. Extract the day-D portion (steps DAY_SKIP .. end)
    y_day = y_pred[:, DAY_SKIP:]   # (n, DAY_STEPS=288)

    # 9. Build output DataFrame and collect KPI data
    print("\n[8] Building output DataFrame ...")
    rows          = []
    actual_list   = []
    pred_list     = []

    for i, (d, loc) in enumerate(valid):
        preds = y_day[i]   # 288 values for day D

        for step in range(DAY_STEPS):
            ts       = d + pd.Timedelta(minutes=5 * step)
            pred_val = float(max(0.0, preds[step]))
            rows.append({'DateTime': ts, 'Forecast_Consumption_kW': round(pred_val, 1)})

            # Actual: index in df = loc (midnight D-1) + DAY_SKIP + step
            actual_idx = loc + DAY_SKIP + step
            if actual_idx < len(load_values):
                av = float(load_values[actual_idx])
                if np.isfinite(av) and av > 0:
                    actual_list.append(av)
                    pred_list.append(pred_val)

    out_df = pd.DataFrame(rows).sort_values('DateTime').reset_index(drop=True)

    # 10. Full-year KPIs
    print("\n" + "=" * 65)
    print(f"  FULL-YEAR KPIs  —  {TARGET_YEAR}  (day-ahead consumption forecast)")
    print("=" * 65)

    if actual_list:
        actual_arr = np.array(actual_list, dtype=np.float32)
        pred_arr   = np.array(pred_list,   dtype=np.float32)
        m          = compute_metrics(actual_arr, pred_arr)

        coverage = len(actual_list) / len(rows) * 100
        bias     = float(pred_arr.mean() - actual_arr.mean())

        print(f"  Days predicted          : {n}")
        print(f"  Steps (5-min) total     : {len(rows):,}")
        print(f"  Steps with actual data  : {len(actual_list):,}  ({coverage:.1f} %)")
        print(f"  -----------------------------------------------------")
        print(f"  RMSE                    : {m['rmse']:.2f} kW")
        print(f"  MAE                     : {m['mae']:.2f} kW")
        print(f"  MAPE                    : {m['mape']:.2f} %")
        print(f"  Mean bias (pred-actual) : {bias:+.2f} kW")
        print(f"  Mean forecast           : {pred_arr.mean():.1f} kW")
        print(f"  Mean actual             : {actual_arr.mean():.1f} kW")
        print(f"  Max forecast            : {pred_arr.max():.1f} kW")
        print(f"  Max actual              : {actual_arr.max():.1f} kW")

        # Monthly breakdown
        out_df['Month'] = out_df['DateTime'].dt.month
        months = out_df['DateTime'].dt.month.values[:len(pred_list)]
        print(f"\n  Monthly RMSE / MAE / MAPE")
        print(f"  {'Month':<10} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}  {'N':>6}")
        print(f"  {'-'*44}")
        month_idx_list = []
        for step_i, (d, loc) in enumerate(valid):
            for s in range(DAY_STEPS):
                actual_idx = loc + DAY_SKIP + s
                if actual_idx < len(load_values):
                    av = float(load_values[actual_idx])
                    if np.isfinite(av) and av > 0:
                        month_idx_list.append(d.month)

        month_idx_arr = np.array(month_idx_list)
        for mo in range(1, 13):
            mask = month_idx_arr == mo
            if mask.sum() == 0:
                continue
            mm = compute_metrics(actual_arr[mask], pred_arr[mask])
            mo_name = pd.Timestamp(f'2025-{mo:02d}-01').strftime('%b')
            print(f"  {mo_name:<10} {mm['rmse']:>8.1f} {mm['mae']:>8.1f} {mm['mape']:>7.1f}%  {mask.sum():>6,}")
        print("=" * 65)
    else:
        print("  WARNING: No actual Load_Is data found — KPIs cannot be computed.")
        print("  (Data for 2025 may not be in Data_Prediction.xlsx yet.)")

    # 11. Backfill early-January days that lacked enough lookback
    #     Jan 1-2 are holidays → use 2 weeks later (same weekday, avoids holiday bias)
    #     Jan 3-8  → use 1 week later (same weekday)
    out_df = out_df.drop(columns=['Month'], errors='ignore')

    BACKFILL_MAP = {
        pd.Timestamp('2025-01-01'): pd.Timestamp('2025-01-15'),
        pd.Timestamp('2025-01-02'): pd.Timestamp('2025-01-16'),
        pd.Timestamp('2025-01-03'): pd.Timestamp('2025-01-10'),
        pd.Timestamp('2025-01-04'): pd.Timestamp('2025-01-11'),
        pd.Timestamp('2025-01-05'): pd.Timestamp('2025-01-12'),
        pd.Timestamp('2025-01-06'): pd.Timestamp('2025-01-13'),
        pd.Timestamp('2025-01-07'): pd.Timestamp('2025-01-14'),
        pd.Timestamp('2025-01-08'): pd.Timestamp('2025-01-15'),
    }

    backfill_rows = []
    out_df['_date'] = out_df['DateTime'].dt.normalize()

    for target_day, ref_day in BACKFILL_MAP.items():
        ref_rows = out_df[out_df['_date'] == ref_day].copy()
        if ref_rows.empty:
            print(f"  WARNING: reference day {ref_day.date()} not found — skipping {target_day.date()}")
            continue
        ref_rows['DateTime'] = ref_rows['DateTime'].apply(
            lambda dt: target_day + pd.Timedelta(hours=dt.hour, minutes=dt.minute)
        )
        ref_rows['_date'] = target_day
        backfill_rows.append(ref_rows)
        print(f"  Backfill {target_day.date()} <- {ref_day.date()}")

    out_df = out_df.drop(columns=['_date'])
    if backfill_rows:
        extra = pd.concat(backfill_rows, ignore_index=True).drop(columns=['_date'])
        out_df = pd.concat([extra, out_df], ignore_index=True).sort_values('DateTime').reset_index(drop=True)

    # 12. Save CSV
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved : {OUTPUT_PATH}")
    print(f"  Rows     : {len(out_df):,}")
    print(f"  From     : {out_df['DateTime'].iloc[0]}")
    print(f"  To       : {out_df['DateTime'].iloc[-1]}")
    print(f"  Mean Forecast_Consumption_kW : {out_df['Forecast_Consumption_kW'].mean():.1f} kW")


if __name__ == '__main__':
    main()