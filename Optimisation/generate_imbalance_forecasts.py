"""
generate_imbalance_forecasts.py
===============================
One-time generator: builds Input/imbalance_forecasts.xlsx with five BG-price
forecast variants used by the value-of-information study.

Variants
--------
  REAL          : actual realised prices (perfect foresight benchmark)
  ROLLING_MEAN  : 7-day rolling mean shifted 1 day forward (today's operational
                  forecast; empirical R^2 ~ 0.06, measured below)
  R2_02         : synthetic AR(1) noise calibrated to R^2 = 0.2
  R2_05         : synthetic AR(1) noise calibrated to R^2 = 0.5
  R2_08         : synthetic AR(1) noise calibrated to R^2 = 0.8

Synthesis methodology
---------------------
  forecast[t] = real[t] + eps[t]
  eps[t]      = rho * eps[t-1] + sqrt(1 - rho^2) * N(0, sigma_eps)
  with sigma_eps^2 = (1 - R^2_target) * Var(real_centred)
  and rho measured empirically as lag-1 autocorrelation of the residuals
  (real - CURRENT). Random seed fixed at 42.

Metrics reported per variant
----------------------------
  R^2, MAE, RMSE, Pinball loss at q=0.9 (penalises under-predicting BG_short
  spikes), Sign accuracy (vs median), Spike F1 (BG_short > 200 EUR/MWh).

Output : Input/imbalance_forecasts.xlsx
         Sheet 'Forecasts' = one column per (variant, side)
         Sheet 'Metrics'   = numeric scorecard

Run once:
  cd Optimisation
  python generate_imbalance_forecasts.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, 'Input')
MATIS_PATH = os.path.join(INPUT_DIR, 'matis_2025_.xlsx')
OUT_PATH   = os.path.join(INPUT_DIR, 'imbalance_forecasts.xlsx')

SEED                 = 42
TARGET_R2_VARIANTS   = [0.2, 0.5, 0.8]
ROLLING_WINDOW_STEPS = 2016   # 7 days * 288 steps/day at 5-min resolution
LOOKAHEAD_SHIFT      = 288    # 1 day forward (no look-ahead)
SPIKE_THRESHOLD_EUR  = 200.0  # for spike-detection F1 on BG_Short


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_real_series():
    """Read DateTime + real BG_Long / BG_Short from matis_2025_.xlsx."""
    raw = pd.read_excel(MATIS_PATH, header=None, skiprows=3, engine='calamine')
    df = raw.iloc[:, [0, 19, 20]].copy()
    df.columns = ['DateTime', 'BG_Long_EUR_MWh', 'BG_Short_EUR_MWh']
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)
    for col in ['BG_Long_EUR_MWh', 'BG_Short_EUR_MWh']:
        df[col] = pd.to_numeric(df[col], errors='coerce').interpolate('linear').bfill().ffill()
    return df


# ----------------------------------------------------------------------------
# Forecast variants
# ----------------------------------------------------------------------------

def current_rolling_mean(real):
    """The 7-day rolling mean shifted 1 day forward (matches load_data())."""
    return (real.rolling(ROLLING_WINDOW_STEPS, min_periods=48)
                .mean()
                .shift(LOOKAHEAD_SHIFT)
                .bfill().ffill())


def synthetic_ar1(real, target_r2, rho, seed):
    """Generate a forecast = real + AR(1) noise scaled to hit target R^2.

    R^2 = 1 - SSE/SST = 1 - Var(real - forecast) / Var(real - mean(real))
        = 1 - sigma_eps^2 / sigma_real^2  (when eps is mean-0 and indep. of real)
    so sigma_eps^2 = (1 - R^2) * sigma_real^2.
    """
    real_arr = np.asarray(real, dtype=float)
    real_centred_var = float(np.var(real_arr - real_arr.mean()))
    sigma_eps_sq = (1.0 - target_r2) * real_centred_var
    sigma_eps    = np.sqrt(sigma_eps_sq)
    sigma_eta    = sigma_eps * np.sqrt(1.0 - rho ** 2)

    rng     = np.random.default_rng(seed)
    n       = len(real_arr)
    epsilon = np.empty(n)
    # initialise at stationary distribution
    epsilon[0] = rng.normal(0.0, sigma_eps)
    for t in range(1, n):
        epsilon[t] = rho * epsilon[t - 1] + rng.normal(0.0, sigma_eta)
    return real_arr + epsilon


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def r_squared(actual, forecast):
    a, f = np.asarray(actual), np.asarray(forecast)
    sse = float(np.sum((a - f) ** 2))
    sst = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan


def mae(actual, forecast):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def pinball_loss(actual, forecast, q=0.9):
    """Pinball loss at quantile q. Under-prediction (forecast < actual) is
    penalised by q; over-prediction by (1-q). For BG_Short, q=0.9 punishes
    missing the upper tail (the expensive spikes)."""
    diff = np.asarray(actual) - np.asarray(forecast)
    losses = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(losses.mean())


def sign_accuracy_vs_median(actual, forecast):
    """% of steps where forecast and actual lie on the same side of the median."""
    a, f = np.asarray(actual), np.asarray(forecast)
    med  = np.median(a)
    return float(np.mean((a > med) == (f > med)))


def spike_f1(actual, forecast, threshold=SPIKE_THRESHOLD_EUR):
    a_sp = np.asarray(actual)   > threshold
    f_sp = np.asarray(forecast) > threshold
    tp = int(np.sum(a_sp & f_sp))
    fp = int(np.sum(~a_sp & f_sp))
    fn = int(np.sum(a_sp & ~f_sp))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def all_metrics(name, actual, forecast):
    return {
        'Variant':        name,
        'R2':             r_squared(actual, forecast),
        'MAE_EUR_MWh':    mae(actual, forecast),
        'RMSE_EUR_MWh':   rmse(actual, forecast),
        'Pinball_q0.9':   pinball_loss(actual, forecast, q=0.9),
        'Sign_Accuracy':  sign_accuracy_vs_median(actual, forecast),
        'Spike_F1':       spike_f1(actual, forecast),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print('Loading real BG series from', MATIS_PATH)
    df = load_real_series()
    n  = len(df)
    print(f'  rows: {n:,}   range: {df["DateTime"].min()} -> {df["DateTime"].max()}')

    real_l = df['BG_Long_EUR_MWh']
    real_s = df['BG_Short_EUR_MWh']

    # ---- CURRENT (7-day rolling mean, shifted 1 day) -----------------------
    cur_l = current_rolling_mean(real_l)
    cur_s = current_rolling_mean(real_s)

    # ---- Measure AR(1) rho from residuals (real - CURRENT) ----------------
    rho_l = float(pd.Series(real_l - cur_l).autocorr(lag=1))
    rho_s = float(pd.Series(real_s - cur_s).autocorr(lag=1))
    print(f'\nMeasured AR(1) rho from residuals:')
    print(f'  BG_Long  : rho = {rho_l:.4f}')
    print(f'  BG_Short : rho = {rho_s:.4f}')

    # ---- Synthetic AR(1) variants (independent BG_Long / BG_Short noise) --
    forecasts = {
        'Real':         (real_l, real_s),
        'Rolling_Mean': (cur_l,  cur_s),
    }
    for target_r2 in TARGET_R2_VARIANTS:
        tag = f'R2_{int(round(target_r2*100)):02d}'
        # independent seeds for the two sides (still derived from SEED for reproducibility)
        seed_l = SEED                  + int(target_r2 * 100)
        seed_s = SEED + 1_000_000      + int(target_r2 * 100)
        f_l = synthetic_ar1(real_l, target_r2, rho_l, seed_l)
        f_s = synthetic_ar1(real_s, target_r2, rho_s, seed_s)

        # Enforce the physical market rule BG_Long <= BG_Short at every step.
        # Real BG prices always satisfy this (it costs more to under-deliver than
        # to over-deliver). Synthetic noise can violate it, which makes the LP
        # objective unbounded — clip the violations.
        violations = (f_l > f_s).sum()
        if violations > 0:
            print(f'  {tag}: clipping {violations:,} steps where BG_Long > BG_Short')
            f_l = np.minimum(f_l, f_s)

        forecasts[tag] = (pd.Series(f_l, index=df.index),
                          pd.Series(f_s, index=df.index))

    # ---- Build Forecasts sheet -------------------------------------------
    out = pd.DataFrame({'DateTime': df['DateTime']})
    for tag, (f_l, f_s) in forecasts.items():
        out[f'BG_Long_{tag}_EUR_MWh']  = f_l.values
        out[f'BG_Short_{tag}_EUR_MWh'] = f_s.values

    # ---- Build Metrics sheet ---------------------------------------------
    metric_rows = []
    for tag, (f_l, f_s) in forecasts.items():
        metric_rows.append({'Side': 'BG_Long',  **all_metrics(tag, real_l.values, f_l.values)})
        metric_rows.append({'Side': 'BG_Short', **all_metrics(tag, real_s.values, f_s.values)})
    metrics_df = pd.DataFrame(metric_rows)

    print('\n=== Metrics summary ===')
    pd.set_option('display.float_format', lambda v: f'{v:8.3f}')
    print(metrics_df.to_string(index=False))

    # ---- Save -----------------------------------------------------------
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f'\nWriting {OUT_PATH} ...')
    with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
        out.to_excel(writer, sheet_name='Forecasts', index=False)
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    print('Done.')


if __name__ == '__main__':
    main()