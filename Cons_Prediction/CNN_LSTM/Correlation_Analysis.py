"""
Correlation_Analysis.py
=======================
Seasonal Pearson correlation matrices between electricity consumption
and model input features. Produces one matrix per season (Winter,
Spring, Summer, Autumn) in a publication-ready 2x2 layout.

Usage
-----
    python Correlation_Analysis.py
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Suppress TensorFlow startup messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'Output Forecast',
                           'Correlation_Seasonal.png')

sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Load data (reuses existing pipeline)
# ---------------------------------------------------------------------------
print("Loading data pipeline  (TensorFlow initialisation may take ~30 s) ...")
from CNN_LSTM_Prediction_TEST import load_data, load_pv_nn_model

pv_nn = load_pv_nn_model()
df    = load_data(pv_nn=pv_nn)

print(f"  Dataset: {len(df):,} rows  |  "
      f"{df['DateTime'].min().date()} → {df['DateTime'].max().date()}")

# ---------------------------------------------------------------------------
# Derive simple calendar columns
# ---------------------------------------------------------------------------
df['Hour']  = df['DateTime'].dt.hour
df['Month'] = df['DateTime'].dt.month

# ---------------------------------------------------------------------------
# Feature mapping  (raw column → readable label)
# Circular sin/cos encodings are excluded — Pearson is not meaningful for them.
# Raw Hour and Weekday (0-6) are used instead.
# ---------------------------------------------------------------------------
FEATURES = {
    'Consumption (kW)':   'Load_Is',
    'Lagged Load −24h':   'Load_yesterday',
    'Lagged Load −7d':    'Load_last_week',
    'PV Estimate (kW)':   'PV_Est',
    'Temperature (°C)':   'Temp_Forecast',
    'Irradiance (W/m²)':  'Irr_FC',
    'Cloud Cover (%)':    'Cloud_Cover',
    'Rain (mm)':          'Rain_Forecast',
    'Hour of Day':        'Hour',
    'Day of Week':        'Weekday',
    'Public Holiday':     'PHolyday',
}

# Keep only columns present in the loaded DataFrame
FEATURES = {label: col for label, col in FEATURES.items() if col in df.columns}
labels   = list(FEATURES.keys())
cols     = list(FEATURES.values())

# ---------------------------------------------------------------------------
# Season definition
# ---------------------------------------------------------------------------
SEASONS = {
    'Winter  (Dec – Feb)': [12, 1, 2],
    'Spring  (Mar – May)': [3, 4, 5],
    'Summer  (Jun – Aug)': [6, 7, 8],
    'Autumn  (Sep – Nov)': [9, 10, 11],
}

# ---------------------------------------------------------------------------
# Plot — 2×2 grid, one heatmap per season
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    9,
    'axes.linewidth': 0.8,
})

fig, axes = plt.subplots(2, 2, figsize=(18, 15))
fig.suptitle(
    'Seasonal Pearson Correlation Matrix — Electricity Consumption vs. Model Features',
    fontsize=13, fontweight='bold', y=1.01,
)

for ax, (season, months) in zip(axes.flat, SEASONS.items()):
    sub  = df.loc[df['Month'].isin(months), cols].dropna()
    corr = sub.corr()
    corr.index   = labels
    corr.columns = labels

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True   # hide upper triangle

    sns.heatmap(
        corr,
        ax          = ax,
        mask        = mask,
        annot       = True,
        fmt         = '.2f',
        cmap        = 'RdBu_r',
        center      = 0,
        vmin        = -1,
        vmax        =  1,
        square      = True,
        linewidths  = 0.4,
        linecolor   = '#cccccc',
        annot_kws   = {'size': 7.5},
        cbar_kws    = {'shrink': 0.75, 'label': 'Pearson r'},
    )

    n = len(sub)
    ax.set_title(f'{season}   (n = {n:,} observations)',
                 fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(axis='x', labelrotation=40, labelsize=8)
    ax.tick_params(axis='y', labelrotation=0,  labelsize=8)

plt.tight_layout(h_pad=4, w_pad=3)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {OUTPUT_PATH}")
plt.show()