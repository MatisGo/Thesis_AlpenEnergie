"""
generate_consumption_forecasts.py
=================================
One-time generator: builds Input/consumption_forecasts.xlsx with five
consumption-forecast variants for the value-of-information study.

Variants
--------
  PERFECT  : Forecast = real consumption (zero error, theoretical upper bound)
  CURRENT  : The existing CNN-LSTM forecast (already in matis_2025_.xlsx)
  MAPE_10  : Synthetic forecast with target MAPE ~ 10%
  MAPE_15  : Synthetic forecast with target MAPE ~ 15%
  MAPE_20  : Synthetic forecast with target MAPE ~ 20%

Synthesis methodology - hour-of-day-conditional additive AR(1) noise
--------------------------------------------------------------------
Real load forecast errors:
  (a) are larger during day than at night    -> hour-of-day-varying variance
  (b) cluster in time                         -> AR(1) autocorrelation
  (c) tend to scale with load level — captured automatically because daytime
      (high-load) absolute errors are larger than nighttime ones.

We use additive noise with hour-of-day-conditional std (in kW, not %), which is
more robust than multiplicative when consumption occasionally dips near zero
(otherwise relative errors explode).

  1. Measure empirical absolute-error std per hour from CNN-LSTM residuals:
        sigma_abs(h) [kW]  for h = 0 .. 23
  2. Measure AR(1) lag-1 autocorrelation rho of the residuals.
  3. For each target WMAPE, generate
         z[t] = rho * z[t-1] + sqrt(1-rho^2) * N(0, 1)     (stationary, unit var.)
         eps[t] = scale * sigma_abs(hour(t)) * z[t]
         forecast[t] = max(0, real[t] + eps[t])
     where scale calibrated so that target WMAPE is hit:
         WMAPE = mean|err| / mean|real|
               = scale * sqrt(2/pi) * mean_t(sigma_abs(hour)) / mean(real)
         -> scale = target_WMAPE * mean(real) / (sqrt(2/pi) * mean_h sigma_abs(h))

Independent random seeds per variant (derived from SEED for reproducibility).

Why WMAPE not per-timestep MAPE
-------------------------------
  Per-timestep MAPE = mean(|err| / actual) blows up at low-consumption hours
  (a 100 kW error at 200 kW load = 50%). Industry standard for load forecasting
  is WMAPE = sum(|err|) / sum(actual), which matches our CNN-LSTM "5% MAPE"
  reference (MAE ~ 89 kW / mean ~ 1778 kW ~ 5%).

Metrics reported per variant (Metrics sheet)
--------------------------------------------
  R2, MAE [kW], RMSE [kW], WMAPE [%] (headline), per-timestep MAPE [%]
  for completeness, Sign accuracy vs median, Spike F1.
  (Spike defined as consumption > top-20% quantile of the real series.)

Output : Input/consumption_forecasts.xlsx
         Sheet 'Forecasts' = one column per variant
         Sheet 'Metrics'   = numeric scorecard

Run once:
  cd Optimisation
  python generate_consumption_forecasts.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, 'Input')
MATIS_PATH = os.path.join(INPUT_DIR, 'matis_2025_.xlsx')
OUT_PATH   = os.path.join(INPUT_DIR, 'consumption_forecasts.xlsx')

SEED                  = 42
TARGET_WMAPE_VARIANTS = [0.10, 0.15, 0.20]      # 10%, 15%, 20% WMAPE
SPIKE_QUANTILE        = 0.80                    # top 20% consumption = "spike"


# ----------------------------------------------------------------------------
# Load real consumption + existing CNN-LSTM forecast
# ----------------------------------------------------------------------------

def load_real_series():
    raw = pd.read_excel(MATIS_PATH, header=None, skiprows=3, engine='calamine')
    # col 0 DateTime, col 6 Consumption_kW, col 21 Forecast_Consumption_kW
    df = raw.iloc[:, [0, 6, 21]].copy()
    df.columns = ['DateTime', 'Consumption_kW', 'Forecast_Consumption_kW']
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)
    for c in ['Consumption_kW', 'Forecast_Consumption_kW']:
        df[c] = pd.to_numeric(df[c], errors='coerce').interpolate('linear').bfill().ffill()
    return df


# ----------------------------------------------------------------------------
# Synthesis
# ----------------------------------------------------------------------------

def synthesize_consumption(real, sigma_abs_by_hour, rho, scale, hours, seed):
    """forecast = max(0, real + eps), eps = scale * sigma_abs(hour) * AR(1)_unit."""
    n   = len(real)
    rng = np.random.default_rng(seed)
    # Stationary unit-variance AR(1)
    z = np.empty(n)
    z[0] = rng.normal(0.0, 1.0)
    drive_std = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + drive_std * rng.normal(0.0, 1.0)
    sigma_t  = sigma_abs_by_hour[hours]                # broadcast sigma_abs by hour
    eps      = scale * sigma_t * z
    forecast = np.maximum(0.0, real + eps)
    return forecast


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def metrics_for_variant(name, actual, forecast):
    a = np.asarray(actual,   dtype=float)
    f = np.asarray(forecast, dtype=float)
    err  = a - f
    sse  = float(np.sum(err ** 2))
    sst  = float(np.sum((a - a.mean()) ** 2))
    r2   = 1.0 - sse / sst if sst > 0 else np.nan
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # WMAPE (industry standard for load forecasting; matches "5% MAPE" headline)
    wmape = float(np.sum(np.abs(err)) / max(np.sum(a), 1.0)) * 100.0
    # Per-timestep MAPE — reported for completeness; inflated at low loads
    a_safe = np.where(a > 1.0, a, np.nan)
    mape_pertstep = float(np.nanmean(np.abs(err) / a_safe)) * 100.0
    sign_acc = float(np.mean((a > np.median(a)) == (f > np.median(a))))
    thr   = float(np.quantile(a, SPIKE_QUANTILE))
    a_sp  = a > thr
    f_sp  = f > thr
    tp = int(np.sum(a_sp & f_sp))
    fp = int(np.sum(~a_sp & f_sp))
    fn = int(np.sum(a_sp & ~f_sp))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        'Variant':           name,
        'R2':                r2,
        'MAE_kW':            mae,
        'RMSE_kW':           rmse,
        'WMAPE_pct':         wmape,            # <- headline (industry-standard)
        'MAPE_pertstep_pct': mape_pertstep,    # <- raw per-step (inflated)
        'Sign_Accuracy':     sign_acc,
        'Spike_F1':          f1,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print(f'Reading {MATIS_PATH}')
    df = load_real_series()
    n  = len(df)
    print(f'  rows: {n:,}   range: {df["DateTime"].min()} -> {df["DateTime"].max()}')

    real  = df['Consumption_kW'].values
    cnn   = df['Forecast_Consumption_kW'].values
    hours = df['DateTime'].dt.hour.values.astype(int)

    # ── Empirical noise structure of the CNN-LSTM forecast ──────────────
    err_abs = real - cnn

    # Time-of-day std of absolute error (in kW)  — clean, no div-by-zero issues
    sigma_abs_by_hour = np.zeros(24)
    for h in range(24):
        mask = hours == h
        if mask.any():
            sigma_abs_by_hour[h] = float(err_abs[mask].std())

    avg_sigma_abs = float(np.mean(sigma_abs_by_hour))
    mean_real     = float(np.mean(real))

    cnn_wmape = float(np.sum(np.abs(err_abs)) / np.sum(real)) * 100.0
    cnn_pertstep_mape = float(np.nanmean(np.abs(err_abs) /
                                      np.where(real > 1.0, real, np.nan))) * 100.0

    print(f'\nCNN-LSTM empirical WMAPE (sum|err|/sum|act|): {cnn_wmape:.2f}%')
    print(f'CNN-LSTM empirical per-timestep MAPE        : {cnn_pertstep_mape:.2f}%  (inflated at low loads)')
    print(f'Mean consumption: {mean_real:.1f} kW')
    print('Absolute error std (sigma_abs) by hour-of-day [kW]:')
    for h in range(0, 24, 2):
        print(f'  {h:>2d}h: {sigma_abs_by_hour[h]:>6.1f}    {h+1:>2d}h: {sigma_abs_by_hour[h+1]:>6.1f}')
    print(f'  Average sigma_abs over 24 h = {avg_sigma_abs:.1f} kW')

    rho = float(pd.Series(err_abs).autocorr(lag=1))
    print(f'\nMeasured AR(1) rho from residuals: {rho:.4f}')

    # ── Build variants ─────────────────────────────────────────────────
    variants = {
        'Perfect': real.copy(),
        'Current': cnn.copy(),
    }
    for target_wmape in TARGET_WMAPE_VARIANTS:
        tag   = f'MAPE_{int(round(target_wmape * 100)):02d}'
        # WMAPE = scale * sqrt(2/pi) * mean_h sigma_abs(h) / mean(real)
        scale = (target_wmape * mean_real
                 / (avg_sigma_abs * np.sqrt(2.0 / np.pi)) if avg_sigma_abs > 0 else 1.0)
        seed  = SEED + int(target_wmape * 100)
        fc    = synthesize_consumption(real, sigma_abs_by_hour, rho, scale, hours, seed)
        variants[tag] = fc
        print(f'  {tag}: scale={scale:.3f}  seed={seed}')

    # ── Build sheets ──────────────────────────────────────────────────
    out = pd.DataFrame({'DateTime': df['DateTime']})
    for tag, fc in variants.items():
        out[f'Forecast_Consumption_{tag}_kW'] = fc

    metric_rows = [metrics_for_variant(tag, real, fc) for tag, fc in variants.items()]
    metrics_df  = pd.DataFrame(metric_rows)

    print('\n=== Metrics summary ===')
    pd.set_option('display.float_format', lambda v: f'{v:8.3f}')
    print(metrics_df.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f'\nWriting {OUT_PATH} ...')
    with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as w:
        out.to_excel(w, sheet_name='Forecasts', index=False)
        metrics_df.to_excel(w, sheet_name='Metrics', index=False)
    print('Done.')


if __name__ == '__main__':
    main()