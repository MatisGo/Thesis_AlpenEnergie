"""
backtest_prod.py
================
In-sample backtesting for DNN_Prod_Prediction over a full year.

Uses the already-trained models (daily + shape) to predict every day
in the period and compare against actual values.

Output Excel (Cons_Prediction/Output Forecast/Backtest_Prod_YYYY-MM-DD.xlsx):
    Sheet "Summary"      : Date, Actual_kW, Predicted_kW, Error_kW, Error_%
    Sheet "Distribution" : DateTime (5-min), Actual_kW, Predicted_kW

Command:
    python backtest_prod.py
    python backtest_prod.py --start 2025-02-15 --end 2026-02-15
"""

import os
import sys
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import DNN_Prod_Prediction as DNN

OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'Output Forecast')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SINGLE-DAY PREDICTION HELPERS
# =============================================================================

def _predict_daily_avg(ref_date, df, daily_w, model, scalers):
    w_dict   = DNN._weather_lookup(daily_w)
    target_d = (ref_date + pd.Timedelta(days=1)).date()
    if target_d not in w_dict:
        return None

    sub = df[df['Date'] <= ref_date].reset_index(drop=True)
    if len(sub) <= DNN.N_LAG_DAYS:
        return None

    today_idx = len(sub) - 1
    prod      = sub['prod_total'].values
    lag       = prod[today_idx - DNN.N_LAG_DAYS: today_idx]

    doy = (ref_date + pd.Timedelta(days=1)).dayofyear
    mon = (ref_date + pd.Timedelta(days=1)).month
    cal = np.array([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                    np.sin(2*np.pi*mon/12),      np.cos(2*np.pi*mon/12),
                    w_dict[target_d][0],          w_dict[target_d][1]])
    reserv = np.array([sub['bidmi_last'].values[today_idx],
                       sub['hasel_last'].values[today_idx]])
    x = np.concatenate([lag, cal, reserv]).reshape(1, -1)

    y_sc   = model.predict(scalers['X'].transform(x), verbose=0).ravel()
    y_pred = scalers['y'].inverse_transform(y_sc.reshape(-1, 1)).ravel()[0]
    return float(y_pred)


def _predict_shape(ref_date, daily_avg, df_15min, daily_reservoir, shape_model, shape_scalers):
    target_date = ref_date + pd.Timedelta(days=1)
    target_d    = target_date.date()

    past = sorted(d for d in df_15min['Date'].unique() if d < target_d)
    past = past[-DNN.N_LAG_DAYS_SHAPE:]
    if len(past) < DNN.N_LAG_DAYS_SHAPE:
        return None

    res_lookup = {row['Date'].date(): (row['bidmi_last'], row['hasel_last'])
                  for _, row in daily_reservoir.iterrows()}

    lag_profiles = []
    for d in past:
        slots = df_15min[df_15min['Date'] == d]['prod_15min'].values
        if len(slots) < DNN.SHAPE_STEPS_PER_DAY:
            return None
        d_mean = max(slots[:DNN.SHAPE_STEPS_PER_DAY].mean(), 1.0)
        lag_profiles.append(slots[:DNN.SHAPE_STEPS_PER_DAY].astype(np.float32) / d_mean)

    lag_flat  = np.concatenate(lag_profiles)
    dow       = target_date.dayofweek
    dow_feat  = np.array([np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])
    res       = res_lookup.get(past[-1], (0.0, 0.0))
    reserv    = np.array(res, dtype=np.float32)

    x          = np.concatenate([lag_flat, dow_feat, reserv]).reshape(1, -1)
    shape_sc   = shape_model.predict(shape_scalers['X'].transform(x), verbose=0)
    shape_norm = np.clip(shape_scalers['y'].inverse_transform(shape_sc)[0], 0, None)

    prod_15   = np.clip(shape_norm * daily_avg, 0, DNN.MAX_PRODUCTION_KW)
    t_15      = np.arange(DNN.SHAPE_STEPS_PER_DAY) * 15.0
    t_5       = np.arange(288) * 5.0
    prod_5min = np.clip(np.interp(t_5, t_15, prod_15), 0, DNN.MAX_PRODUCTION_KW)

    base = pd.Timestamp(target_d)
    return pd.DataFrame({
        'DateTime':          [base + pd.Timedelta(minutes=5*i) for i in range(288)],
        'Predicted_kW': np.round(prod_5min, 1),
    })


def _get_actual_distribution(target_date, df_15min):
    d     = target_date.date()
    slots = df_15min[df_15min['Date'] == d]['prod_15min'].values
    if len(slots) < DNN.SHAPE_STEPS_PER_DAY:
        return None
    slots = slots[:DNN.SHAPE_STEPS_PER_DAY].astype(np.float32)
    t_15  = np.arange(DNN.SHAPE_STEPS_PER_DAY) * 15.0
    t_5   = np.arange(288) * 5.0
    act_5 = np.interp(t_5, t_15, slots)
    base  = pd.Timestamp(d)
    return pd.DataFrame({
        'DateTime':  [base + pd.Timedelta(minutes=5*i) for i in range(288)],
        'Actual_kW': np.round(act_5, 1),
    })


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(date_range, df, daily_w, df_15min, daily_res):
    from tensorflow.keras.models import load_model as _load

    print("\n" + "="*60)
    print("  In-sample backtest (existing trained models)")
    print("="*60)

    if not os.path.isfile(DNN.MODEL_PATH) or not os.path.isfile(DNN.SCALER_PATH):
        print("  ERROR: Trained daily model not found. Run --train first.")
        return [], []

    model    = _load(DNN.MODEL_PATH)
    scalers  = joblib.load(DNN.SCALER_PATH)
    has_shape = os.path.isfile(DNN.SHAPE_MODEL_PATH) and os.path.isfile(DNN.SHAPE_SCALER_PATH)
    shape_m  = _load(DNN.SHAPE_MODEL_PATH)    if has_shape else None
    shape_sc = joblib.load(DNN.SHAPE_SCALER_PATH) if has_shape else None

    daily_results, dist_results = [], []
    n = len(date_range)

    for i, ref_date in enumerate(date_range):
        target_date = ref_date + pd.Timedelta(days=1)
        print(f"\r  [{i+1}/{n}] Predicting {target_date.date()} ...", end='', flush=True)

        avg = _predict_daily_avg(ref_date, df, daily_w, model, scalers)
        if avg is None:
            continue
        daily_results.append((ref_date, target_date, avg))

        if shape_m is not None:
            dist = _predict_shape(ref_date, avg, df_15min, daily_res, shape_m, shape_sc)
            if dist is not None:
                dist_results.append((target_date, dist))

    print(f"\n  Done: {len(daily_results)} daily predictions, {len(dist_results)} distributions.")
    return daily_results, dist_results


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def _build_summary(daily_results, df):
    actual_lookup = {row['Date'].date(): row['prod_total'] for _, row in df.iterrows()}
    records = []
    for _, target_date, pred in daily_results:
        d      = target_date.date()
        actual = actual_lookup.get(d, np.nan)
        err    = pred - actual if not np.isnan(actual) else np.nan
        pct    = abs(err / actual * 100) if actual and not np.isnan(err) else np.nan
        records.append({
            'Date':          str(d),
            'Actual_kW':     round(actual, 1) if not np.isnan(actual) else np.nan,
            'Predicted_kW':  round(pred,   1),
            'Error_kW':      round(err,    1) if not np.isnan(err)    else np.nan,
            'Error_%':       round(pct,    1) if not np.isnan(pct)    else np.nan,
        })
    return pd.DataFrame(records)


def _build_distribution(dist_results, df_15min):
    frames = []
    for target_date, pred_df in dist_results:
        actual_df = _get_actual_distribution(target_date, df_15min)
        if actual_df is None:
            continue
        row = actual_df.copy()
        row['Predicted_kW'] = pred_df['Predicted_kW'].values
        frames.append(row)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _kpis(summary):
    mask = summary['Actual_kW'].notna() & summary['Predicted_kW'].notna()
    act  = summary.loc[mask, 'Actual_kW'].values
    pred = summary.loc[mask, 'Predicted_kW'].values
    rmse = np.sqrt(np.mean((pred - act)**2))
    mae  = np.mean(np.abs(pred - act))
    mape = np.mean(np.abs((pred - act) / (act + 1e-6))) * 100
    return rmse, mae, mape


def save_results(summary, distribution):
    today    = datetime.date.today().isoformat()
    out      = os.path.join(OUTPUT_DIR, f'Backtest_Prod_{today}.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        summary.to_excel(      writer, sheet_name='Summary',      index=False)
        distribution.to_excel( writer, sheet_name='Distribution', index=False)
    print(f"\n  Saved: {out}")
    return out


def plot_results(summary):
    summary['Date_dt'] = pd.to_datetime(summary['Date'])

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    ax.plot(summary['Date_dt'], summary['Actual_kW'],    label='Actual',    color='steelblue', lw=1.5)
    ax.plot(summary['Date_dt'], summary['Predicted_kW'], label='Predicted', color='orange',    lw=1, ls='--')
    ax.set_ylabel('Daily Avg Production (kW)')
    ax.set_title('Production Backtest — Actual vs Predicted')
    ax.legend()

    ax2 = axes[1]
    ax2.plot(summary['Date_dt'], summary['Error_kW'].abs(), color='orange', lw=1)
    ax2.axhline(0, color='grey', lw=0.5)
    ax2.set_ylabel('|Error| (kW)')
    ax2.set_xlabel('Date')

    plt.tight_layout()
    plt.show()

    rmse, mae, mape = _kpis(summary)
    print(f"\n{'='*45}")
    print(f"  BACKTEST KPIs")
    print(f"  RMSE : {rmse:,.0f} kW")
    print(f"  MAE  : {mae:,.0f} kW")
    print(f"  MAPE : {mape:.1f} %")
    print(f"{'='*45}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Production Forecast Backtest')
    parser.add_argument('--start', default='2025-02-15', help='Start date YYYY-MM-DD')
    parser.add_argument('--end',   default='2026-02-15', help='End date YYYY-MM-DD')
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start)
    end_date   = pd.Timestamp(args.end)
    print(f"\n  Backtest period: {start_date.date()} → {end_date.date()}")

    print("Loading data ...")
    daily            = DNN.load_daily_production()
    daily_w          = DNN.load_daily_weather()
    df_15min, daily_res = DNN.load_15min_production()
    df               = DNN.merge_and_build_features(daily, daily_w)

    date_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D')
    date_range = [d for d in date_range if d in df['Date'].values]
    print(f"  Reference dates available: {len(date_range)}\n")

    daily_results, dist_results = run_backtest(date_range, df, daily_w, df_15min, daily_res)

    print("\nBuilding result tables ...")
    summary      = _build_summary(daily_results, df)
    distribution = _build_distribution(dist_results, df_15min)

    out_path = save_results(summary, distribution)
    plot_results(summary)
    print(f"\n  Done. Results saved to:\n  {out_path}")


if __name__ == '__main__':
    main()
