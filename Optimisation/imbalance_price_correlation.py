"""
imbalance_price_correlation.py
================================
Same-weekday same-hour lag-correlation analysis for Swissgrid BG-Long
and BG-Short imbalance prices (2025, 5-min resolution).

For each lag W-1 … W-4 (7, 14, 21, 28 days = 2016, 4032, 6048, 8064 steps):
  - Pearson r and R² against the actual series (full year and by calendar month)
  - 4-week average predictor  (mean of W-1 … W-4)

Outputs
-------
  Output/BG_Price_Lag_Correlation.xlsx  — summary + monthly tables
  Output/BG_Price_Lag_Correlation.png   — grouped bar chart (monthly r by lag)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(SCRIPT_DIR, 'Input',  'matis_2025_.xlsx')
OUT_XLSX    = os.path.join(SCRIPT_DIR, 'Output', 'BG_Price_Lag_Correlation.xlsx')
OUT_PNG     = os.path.join(SCRIPT_DIR, 'Output', 'BG_Price_Lag_Correlation.png')

STEPS_PER_DAY  = 288          # 24 h × 12 steps/h
STEPS_PER_WEEK = STEPS_PER_DAY * 7   # 2016

LAG_LABELS = ['W-1', 'W-2', 'W-3', 'W-4', 'Avg W1-4']
LAG_WEEKS  = [1, 2, 3, 4]

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

LAG_COLORS  = ['#2166ac', '#4dac26', '#d01c8b', '#f1a340']   # W-1 … W-4
AVG_COLOR   = '#636363'

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_series():
    raw = pd.read_excel(INPUT_PATH, header=None, skiprows=3, engine='calamine')
    df  = raw.iloc[:, [0, 19, 20]].copy()
    df.columns = ['DateTime', 'BG_Long', 'BG_Short']
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)
    for col in ['BG_Long', 'BG_Short']:
        df[col] = pd.to_numeric(df[col], errors='coerce').interpolate('linear').bfill().ffill()
    return df


# ---------------------------------------------------------------------------
# Pearson r / R² helper
# ---------------------------------------------------------------------------

def pearson_r2(actual, predicted):
    """Return (r, R²) ignoring NaN pairs."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() < 10:
        return np.nan, np.nan
    r, _ = stats.pearsonr(actual[mask], predicted[mask])
    return float(r), float(r ** 2)


# ---------------------------------------------------------------------------
# Build lag series
# ---------------------------------------------------------------------------

def build_lags(series: np.ndarray) -> dict:
    """Return dict lag_label -> lagged array (same length, NaN at start)."""
    n    = len(series)
    lags = {}
    for w in LAG_WEEKS:
        shift = w * STEPS_PER_WEEK
        lagged = np.empty(n)
        lagged[:shift] = np.nan
        lagged[shift:] = series[:n - shift]
        lags[f'W-{w}'] = lagged

    # 4-week average predictor
    stack = np.vstack([lags[f'W-{w}'] for w in LAG_WEEKS])
    lags['Avg W1-4'] = np.nanmean(stack, axis=0)
    # Set to NaN where all four lags are NaN (first 4 weeks)
    all_nan = np.all(np.isnan(stack), axis=0)
    lags['Avg W1-4'][all_nan] = np.nan
    return lags


# ---------------------------------------------------------------------------
# Overall correlation table
# ---------------------------------------------------------------------------

def overall_table(df, lags_long, lags_short):
    rows = []
    for label in LAG_LABELS:
        r_l, r2_l = pearson_r2(df['BG_Long'].values,  lags_long[label])
        r_s, r2_s = pearson_r2(df['BG_Short'].values, lags_short[label])
        rows.append({'Lag': label,
                     'BG_Long_r':  round(r_l,  4),
                     'BG_Long_R2': round(r2_l, 4),
                     'BG_Short_r':  round(r_s,  4),
                     'BG_Short_R2': round(r2_s, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monthly correlation tables
# ---------------------------------------------------------------------------

def monthly_table(df, lags_long, lags_short, metric='r'):
    """metric: 'r' or 'R2'"""
    months = sorted(df['DateTime'].dt.month.unique())
    rows = []
    for m in months:
        mask = df['DateTime'].dt.month == m
        row = {'Month': MONTH_NAMES[m - 1]}
        for label in LAG_LABELS:
            actual_l = df.loc[mask, 'BG_Long'].values
            pred_l   = lags_long[label][mask]
            actual_s = df.loc[mask, 'BG_Short'].values
            pred_s   = lags_short[label][mask]
            r_l, r2_l = pearson_r2(actual_l, pred_l)
            r_s, r2_s = pearson_r2(actual_s, pred_s)
            if metric == 'r':
                row[f'BG_Long_{label}']  = round(r_l,  4)
                row[f'BG_Short_{label}'] = round(r_s,  4)
            else:
                row[f'BG_Long_{label}']  = round(r2_l, 4)
                row[f'BG_Short_{label}'] = round(r2_s, 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grouped bar chart
# ---------------------------------------------------------------------------

def make_chart(df, lags_long, lags_short):
    months      = sorted(df['DateTime'].dt.month.unique())
    month_names = [MONTH_NAMES[m - 1] for m in months]
    n_months    = len(months)

    # Only W-1 to W-4 in the chart (not the average, for clarity)
    plot_lags   = [f'W-{w}' for w in LAG_WEEKS]
    n_lags      = len(plot_lags)
    width       = 0.18
    x           = np.arange(n_months)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle('Monthly Pearson Correlation — Same-Weekday Lag Predictors\n'
                 'Swissgrid Imbalance Prices 2025 (5-min resolution)',
                 fontsize=12, fontweight='bold')

    for ax, series_label, lags, color_base in [
            (axes[0], 'BG-Long',  lags_long,  LAG_COLORS),
            (axes[1], 'BG-Short', lags_short, LAG_COLORS),
    ]:
        for i, (label, color) in enumerate(zip(plot_lags, color_base)):
            r_vals = []
            for m in months:
                mask     = df['DateTime'].dt.month == m
                actual   = df.loc[mask, 'BG_Long' if series_label == 'BG-Long'
                                       else 'BG_Short'].values
                pred     = lags[label][mask]
                r, _     = pearson_r2(actual, pred)
                r_vals.append(r if not np.isnan(r) else 0.0)

            offset = (i - (n_lags - 1) / 2) * width
            bars = ax.bar(x + offset, r_vals, width,
                          label=label, color=color, edgecolor='white', linewidth=0.4)

        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_ylabel('Pearson r', fontsize=10)
        ax.set_title(series_label, fontsize=11, fontweight='bold')
        ax.set_ylim(-0.15, 0.55)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.tick_params(axis='x', labelsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(month_names)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.join(SCRIPT_DIR, 'Output'), exist_ok=True)

    print('Loading BG price data...')
    df = load_series()
    print(f'  {len(df):,} rows  |  {df["DateTime"].min().date()} -> {df["DateTime"].max().date()}')

    print('Building lag series...')
    lags_long  = build_lags(df['BG_Long'].values)
    lags_short = build_lags(df['BG_Short'].values)

    # ── Tables ──────────────────────────────────────────────────────────────
    tbl_overall  = overall_table(df, lags_long, lags_short)
    tbl_monthly_r  = monthly_table(df, lags_long, lags_short, metric='r')
    tbl_monthly_r2 = monthly_table(df, lags_long, lags_short, metric='R2')

    print('\n=== Overall Pearson r and R² ===')
    print(tbl_overall.to_string(index=False))

    # ── Excel ───────────────────────────────────────────────────────────────
    print(f'\nWriting {OUT_XLSX}...')
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        tbl_overall.to_excel(w,    sheet_name='Overall',    index=False)
        tbl_monthly_r.to_excel(w,  sheet_name='Monthly_r',  index=False)
        tbl_monthly_r2.to_excel(w, sheet_name='Monthly_R2', index=False)

    # ── Chart ───────────────────────────────────────────────────────────────
    print(f'Writing {OUT_PNG}...')
    fig = make_chart(df, lags_long, lags_short)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print('\nDone.')


if __name__ == '__main__':
    main()
