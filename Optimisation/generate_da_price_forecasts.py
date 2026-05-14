"""
generate_da_price_forecasts.py
==============================
One-time generator: builds Input/da_price_forecasts.xlsx with three
day-ahead price forecast variants for the value-of-information study.

Variants
--------
  PERFECT  : Forecast = real DA price (zero error, theoretical upper bound)
  MAPE_05  : Additive AR(1) noise calibrated to ~5% MAPE  (best-in-class)
  MAPE_15  : Additive AR(1) noise calibrated to ~15% MAPE (volatile 2022-equivalent)

Synthesis methodology — additive AR(1) noise
---------------------------------------------
Unlike consumption, DA prices can be negative or near-zero, making
multiplicative noise ill-defined.  Additive noise is used instead:

  forecast[t] = real[t] + eps[t]
  eps[t]      = rho * eps[t-1] + sqrt(1 - rho^2) * sigma * N(0, 1)

where sigma is the stationary std of the error process, and the noise
is never clipped (negative forecasts are physically valid for DA prices).

MAPE definition:  MAPE = mean(|eps|) / mean(|real|)
  For a stationary zero-mean AR(1): E[|eps|] = sigma * sqrt(2/pi)
  => sigma = target_MAPE * mean(|real|) / sqrt(2/pi)

AR(1) persistence (rho):
  Estimated empirically from the residuals of a naive 7-day rolling-mean
  DA price forecast shifted 1 day forward (the same baseline used for the
  imbalance price variants).  This gives a realistic temporal correlation
  for day-ahead price forecast errors.

Output: Input/da_price_forecasts.xlsx
  Sheet 'Forecasts' : DateTime + one column per variant
  Sheet 'Metrics'   : MAE, RMSE, MAPE per variant

Run once:
  cd Optimisation
  python generate_da_price_forecasts.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, 'Input')
MATIS_PATH = os.path.join(INPUT_DIR, 'matis_2025_.xlsx')
OUT_PATH   = os.path.join(INPUT_DIR, 'da_price_forecasts.xlsx')

SEED                 = 99
TARGET_MAPE_VARIANTS = [0.05, 0.15]   # 5% and 15%


# ---------------------------------------------------------------------------
# Load real DA price series
# ---------------------------------------------------------------------------

def load_real_series():
    raw = pd.read_excel(MATIS_PATH, header=None, skiprows=3, engine='calamine')
    # col 0 DateTime, col 11 Day_Ahead_Price_EUR_MWh
    df = raw.iloc[:, [0, 11]].copy()
    df.columns = ['DateTime', 'DA_Price_EUR_MWh']
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)
    df['DA_Price_EUR_MWh'] = (
        pd.to_numeric(df['DA_Price_EUR_MWh'], errors='coerce')
          .interpolate('linear').bfill().ffill()
    )
    return df


# ---------------------------------------------------------------------------
# Noise synthesis
# ---------------------------------------------------------------------------

def synthesize_da_price(real, rho, sigma, seed):
    """Additive stationary AR(1) noise: forecast = real + eps."""
    n   = len(real)
    rng = np.random.default_rng(seed)
    eps = np.empty(n)
    eps[0] = rng.normal(0.0, sigma)
    drive_std = np.sqrt(max(0.0, 1.0 - rho ** 2))
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + drive_std * sigma * rng.normal(0.0, 1.0)
    return real + eps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_for_variant(name, actual, forecast):
    a   = np.asarray(actual,   dtype=float)
    f   = np.asarray(forecast, dtype=float)
    err = a - f
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # MAPE = mean(|err|) / mean(|actual|)  — robust to negative prices
    mean_abs_actual = float(np.mean(np.abs(a)))
    mape = (mae / mean_abs_actual * 100.0) if mean_abs_actual > 0 else np.nan
    sse  = float(np.sum(err ** 2))
    sst  = float(np.sum((a - a.mean()) ** 2))
    r2   = 1.0 - sse / sst if sst > 0 else np.nan
    return {
        'Variant':          name,
        'R2':               r2,
        'MAE_EUR_MWh':      mae,
        'RMSE_EUR_MWh':     rmse,
        'MAPE_pct':         mape,
        'Mean_Abs_Price':   mean_abs_actual,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Reading {MATIS_PATH}')
    df = load_real_series()
    n  = len(df)
    print(f'  rows: {n:,}   range: {df["DateTime"].min()} -> {df["DateTime"].max()}')

    real = df['DA_Price_EUR_MWh'].values
    mean_abs_real = float(np.mean(np.abs(real)))
    print(f'  Mean |DA price|: {mean_abs_real:.2f} EUR/MWh')
    print(f'  Min: {real.min():.2f}  Max: {real.max():.2f}  Mean: {real.mean():.2f} EUR/MWh')

    # ── Estimate AR(1) rho from rolling-mean forecast residuals ──────────
    # 7-day rolling mean (2016 × 5-min steps) shifted 1 day (288 steps) forward
    rolling_mean = (
        pd.Series(real)
          .rolling(2016, min_periods=48)
          .mean()
          .shift(288)
          .bfill()
          .ffill()
          .values
    )
    residuals = real - rolling_mean
    rho = float(pd.Series(residuals).autocorr(lag=1))
    # Clamp to a stable range
    rho = float(np.clip(rho, 0.0, 0.98))
    print(f'\n  Estimated AR(1) rho from rolling-mean residuals: {rho:.4f}')

    # ── Build variants ────────────────────────────────────────────────────
    variants = {'Perfect': real.copy()}

    for target_mape in TARGET_MAPE_VARIANTS:
        tag   = f'MAPE_{int(round(target_mape * 100)):02d}'
        # E[|eps|] = sigma * sqrt(2/pi)  =>  sigma = MAPE * mean(|real|) / sqrt(2/pi)
        sigma = target_mape * mean_abs_real / np.sqrt(2.0 / np.pi)
        seed  = SEED + int(target_mape * 100)
        fc    = synthesize_da_price(real, rho, sigma, seed)
        variants[tag] = fc
        print(f'  {tag}: sigma={sigma:.2f} EUR/MWh  seed={seed}')

    # ── Build output DataFrame ────────────────────────────────────────────
    out = pd.DataFrame({'DateTime': df['DateTime']})
    for tag, fc in variants.items():
        out[f'DA_Price_{tag}_EUR_MWh'] = fc

    metric_rows = [metrics_for_variant(tag, real, fc) for tag, fc in variants.items()]
    metrics_df  = pd.DataFrame(metric_rows)

    print('\n=== Metrics summary ===')
    pd.set_option('display.float_format', lambda v: f'{v:8.3f}')
    print(metrics_df.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f'\nWriting {OUT_PATH} ...')
    with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as w:
        out.to_excel(w, sheet_name='Forecasts', index=False)
        metrics_df.to_excel(w, sheet_name='Metrics', index=False)
    print('Done.')


if __name__ == '__main__':
    main()
