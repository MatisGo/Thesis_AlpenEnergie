"""
Build PV Irradiance-to-Power Correction Table
==============================================
Constructs a month x 15-min-slot correction ratio table from two years of
historical PV production and co-located GHI (Global Horizontal Irradiance) data.

Scientific basis:
-----------------
Lorenz et al. (2011) "Regional PV power prediction for improved grid
integration", Progress in Photovoltaics 19(7), DOI: 10.1002/pip.1033

Bacher et al. (2009) "Online short-term solar power forecasting",
Solar Energy 83(10), DOI: 10.1016/j.solener.2009.05.016

Irradiance type:
----------------
This table uses GHI (Global Horizontal Irradiance), which includes both
direct beam and diffuse components. GHI is consistent with the Open-Meteo
shortwave_radiation output used in Imported_Forecast.xlsx, ensuring that
the ratio table and the forecast irradiance are on the same physical basis.

Method:
-------
For each (month m, 15-min slot s) bin, the correction ratio is:

    ratio[m, s] = sum(PV_kW[m,s]) / sum(GHI[m,s])
                  --- only where GHI > IRR_THRESHOLD W/m2 ---

This gives the effective kW-per-(W/m2) conversion factor including:
  - Panel area and installed capacity
  - System losses (inverter, cable, temperature derating)
  - Local topographic shading geometry (mountain valley)
  - Seasonal geometry differences (sun angle, diffuse fraction)

The ratio is intentionally computed per-month to capture the 8× seasonal
variation observed in the Meiringen valley between June and December.

Usage in prediction:
--------------------
    PV_Est[step] = GHI_forecast[step] * ratio[month - 1, slot_15min]

Where:
  - GHI_forecast comes from Imported_Forecast.xlsx (Open-Meteo shortwave_radiation)
  - slot_15min = int(hour * 4 + minute // 15)  (0-95)
  - month is 0-indexed (January = 0)

Output:
-------
PV_Correction_Table.npz  (machine-readable, used by CNN_LSTM_Prediction.py)
PV_Correction_Table.csv  (human-readable, for inspection / thesis appendix)

PV production source:
---------------------
File: PV_Production_Forecast_23_24.xlsx
  - Period: 2023-01-01 to 2025-01-01 (2 full calendar years)
  - Resolution: 15 minutes (96 slots/day)
  - Columns: Datum, Zeit, 'PV Production [kW]', GHI (W/m2)
  - Max observed PV production: ~754 kW (July peak)
  - Monthly peak means: June ~96.5 kW, December ~12 kW (ratio 8.0x)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../PV_Correction/
PV_FILE    = os.path.join(SCRIPT_DIR, '..', '..', 'PV_Production_Forecast_23_24.xlsx')
OUT_NPZ    = os.path.join(SCRIPT_DIR, 'PV_Correction_Table.npz')
OUT_CSV    = os.path.join(SCRIPT_DIR, 'PV_Correction_Table.csv')
OUT_PLOT   = os.path.join(SCRIPT_DIR, 'PV_Correction_Table_Analysis.png')

# Minimum irradiance threshold to include a sample in the ratio computation.
# Below this value the signal-to-noise ratio is poor (dawn/dusk, heavy overcast).
# Lorenz (2011) recommends excluding sub-threshold irradiance samples.
IRR_THRESHOLD = 10.0   # W/m²

# Minimum number of valid samples per (month, slot) bin required to compute
# a reliable ratio. Bins with fewer samples get value 0 (no PV expected).
MIN_SAMPLES = 5

MONTHS = 12   # 1–12
SLOTS  = 96   # 15-min slots per day (0 = 00:00, 95 = 23:45)

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# =============================================================================
# LOAD & PARSE PV DATA
# =============================================================================

def load_pv_data(path):
    """
    Load the PV production + irradiance forecast file.

    Expected columns (from PV_Production_Forecast_23_24.xlsx):
        Datum   – date (Excel serial or date string)
        Zeit    – time of day (e.g. '00:00:00' or timedelta)
        PV Production [kW]  – measured PV output in kW
        Forecast            – irradiance forecast in W/m²

    Returns
    -------
    df : pd.DataFrame
        Columns: DateTime (15-min), PV_kW, GHI
    """
    print(f"Loading PV data from: {path}")
    df = pd.read_excel(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # --- Parse DateTime ---
    date_col = 'Datum'
    time_col = 'Zeit'

    date_part = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    time_part = pd.to_timedelta(df[time_col].astype(str), errors='coerce')
    df['DateTime'] = date_part + time_part

    # --- Identify PV production column (flexible naming) ---
    pv_col = [c for c in df.columns if 'PV' in c and 'kW' in c]
    if not pv_col:
        pv_col = [c for c in df.columns if 'production' in c.lower()]

    # --- Identify GHI irradiance column ---
    # Accepts: GHI, Global, Horizontal, shortwave, radiation, Irradiance
    # Also accepts old naming: Forecast, forecast (backwards compatible)
    irr_keywords = ['GHI', 'Global', 'Horizontal', 'shortwave', 'radiation',
                    'Irradiance', 'Forecast']
    irr_col = [c for c in df.columns
               if any(kw.lower() in c.lower() for kw in irr_keywords)
               and c not in (pv_col or [])]

    if not pv_col or not irr_col:
        print("ERROR: Could not identify PV Production and GHI columns.")
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError("Check column names in PV file.")

    pv_col  = pv_col[0]
    irr_col = irr_col[0]
    print(f"  Using: PV='{pv_col}', GHI='{irr_col}'")

    df['PV_kW'] = pd.to_numeric(df[pv_col],  errors='coerce').fillna(0.0)
    df['GHI']   = pd.to_numeric(df[irr_col], errors='coerce').fillna(0.0)

    # --- Keep only rows with valid DateTime ---
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    # --- Extract time features ---
    df['Month']     = df['DateTime'].dt.month          # 1–12
    df['HourFrac']  = df['DateTime'].dt.hour + df['DateTime'].dt.minute / 60.0
    df['Slot_15min'] = (df['DateTime'].dt.hour * 4 +
                        df['DateTime'].dt.minute // 15).astype(int)  # 0–95

    print(f"  Date range: {df['DateTime'].min()} → {df['DateTime'].max()}")
    print(f"  PV range:   [{df['PV_kW'].min():.1f}, {df['PV_kW'].max():.1f}] kW")
    print(f"  GHI range: [{df['GHI'].min():.1f}, {df['GHI'].max():.1f}] W/m²")

    return df


# =============================================================================
# BUILD CORRECTION TABLE
# =============================================================================

def build_correction_table(df, irr_threshold=IRR_THRESHOLD, min_samples=MIN_SAMPLES):
    """
    Compute ratio[month, slot] = sum(PV_kW) / sum(GHI)
    for valid daytime samples only (GHI > threshold).

    Using sum/sum rather than mean(PV)/mean(Irr) avoids noise from
    partially-cloudy samples where the ratio is ill-conditioned.
    This is the approach recommended by Lorenz et al. (2011).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_pv_data().
    irr_threshold : float
        Minimum GHI to include (W/m²).
    min_samples : int
        Minimum samples per bin to compute a ratio.

    Returns
    -------
    ratio_table : np.ndarray, shape (12, 96)
        [month-1, slot] → kW / (W/m²) conversion factor.
    n_table : np.ndarray, shape (12, 96)
        Number of valid samples per bin.
    """
    ratio_table = np.zeros((MONTHS, SLOTS), dtype=np.float32)
    n_table     = np.zeros((MONTHS, SLOTS), dtype=np.int32)

    # Filter: both irradiance and PV must be meaningful
    mask = (df['GHI'] > irr_threshold) & (df['PV_kW'] >= 0)
    df_valid = df[mask].copy()

    print(f"\n  Valid samples (GHI > {irr_threshold} W/m²): "
          f"{len(df_valid)} / {len(df)} ({100*len(df_valid)/len(df):.1f}%)")

    for m in range(1, MONTHS + 1):
        for s in range(SLOTS):
            sub = df_valid[(df_valid['Month'] == m) & (df_valid['Slot_15min'] == s)]
            n   = len(sub)
            n_table[m - 1, s] = n

            if n >= min_samples and sub['GHI'].sum() > 0:
                # sum(PV_kW) / sum(GHI)  [kW / (W/m²)]
                ratio_table[m - 1, s] = sub['PV_kW'].sum() / sub['GHI'].sum()
            # else: ratio stays 0 (night/winter slots with no reliable data)

    # Summary statistics
    nonzero = ratio_table[ratio_table > 0]
    print(f"\n  Correction table computed:")
    print(f"    Non-zero bins: {len(nonzero)} / {MONTHS * SLOTS}")
    print(f"    Ratio range: [{nonzero.min():.4f}, {nonzero.max():.4f}] kW/(W/m²)")
    print(f"    Ratio mean (daytime): {nonzero.mean():.4f}")
    print(f"    Monthly peak ratios:")
    for m in range(MONTHS):
        row = ratio_table[m]
        if row.max() > 0:
            print(f"      {MONTH_NAMES[m]}: max ratio = {row.max():.4f} "
                  f"  (peak slot {row.argmax():2d} = "
                  f"{row.argmax() * 15 // 60:02d}:{row.argmax() * 15 % 60:02d})")

    return ratio_table, n_table


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(ratio_table, n_table):
    """Save correction table to .npz and .csv."""

    # 1. NPZ (used by CNN_LSTM_Prediction.py)
    np.savez(OUT_NPZ,
             ratio_table=ratio_table,
             n_table=n_table,
             months=np.arange(1, MONTHS + 1),
             slots=np.arange(SLOTS),
             irr_threshold=np.array([IRR_THRESHOLD]),
             min_samples=np.array([MIN_SAMPLES]))
    print(f"\n  Correction table saved to: {OUT_NPZ}")

    # 2. CSV (human-readable)
    # Row = month (1-12), Column = slot (00:00, 00:15, ..., 23:45)
    slot_labels = [f'{s * 15 // 60:02d}:{s * 15 % 60:02d}' for s in range(SLOTS)]
    csv_df = pd.DataFrame(ratio_table,
                          index=[f'Month_{m:02d}' for m in range(1, MONTHS + 1)],
                          columns=slot_labels)
    csv_df.index.name = 'Month'
    csv_df.to_csv(OUT_CSV)
    print(f"  Correction table saved to: {OUT_CSV}")
    print(f"  Table shape: {ratio_table.shape}  (months × 15-min slots)")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correction_analysis(df, ratio_table, n_table):
    """
    Create a 4-panel analysis figure:
      1. Correction ratio heatmap (month x slot)
      2. Monthly peak PV production profiles (daytime only)
      3. Sample count heatmap (data coverage)
      4. Seasonal variation: max daily ratio per month
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('PV Irradiance-to-Power Correction Table Analysis\n'
                 '(Lorenz 2011 / Bacher 2009 approach)',
                 fontsize=14, fontweight='bold')

    # --- Panel 1: Ratio heatmap ---
    ax1 = axes[0, 0]
    im1 = ax1.imshow(ratio_table, aspect='auto', cmap='YlOrRd',
                     vmin=0, vmax=ratio_table.max())
    plt.colorbar(im1, ax=ax1, label='kW / (W/m²)')
    ax1.set_title('Correction Ratio Table: ratio[month, slot]', fontsize=12, fontweight='bold')
    ax1.set_xlabel('15-min Slot (0 = 00:00)')
    ax1.set_ylabel('Month')
    ax1.set_yticks(range(MONTHS))
    ax1.set_yticklabels(MONTH_NAMES)
    # Mark daytime slots (6h–18h = slots 24–72)
    ax1.axvline(x=24, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=72, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(24, -0.8, '06:00', color='white', fontsize=7, ha='center')
    ax1.text(72, -0.8, '18:00', color='white', fontsize=7, ha='center')

    # --- Panel 2: Monthly PV profiles ---
    ax2 = axes[0, 1]
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / (MONTHS - 1)) for i in range(MONTHS)]
    for m in range(MONTHS):
        profile = ratio_table[m]
        daytime = profile[24:73]  # 06:00–18:00
        slots_dt = np.arange(24, 73)
        if daytime.max() > 0:
            ax2.plot(slots_dt, daytime, color=colors[m], label=MONTH_NAMES[m],
                     linewidth=1.5, alpha=0.85)
    ax2.set_title('Monthly Daytime Correction Profiles', fontsize=12, fontweight='bold')
    ax2.set_xlabel('15-min Slot')
    ax2.set_ylabel('Ratio [kW / (W/m²)]')
    ax2.legend(ncol=2, fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    # Slot labels
    slot_ticks = [24, 32, 40, 48, 56, 64, 72]
    slot_labels_ticks = [f'{s * 15 // 60:02d}:00' for s in slot_ticks]
    ax2.set_xticks(slot_ticks)
    ax2.set_xticklabels(slot_labels_ticks)

    # --- Panel 3: Sample count heatmap ---
    ax3 = axes[1, 0]
    im3 = ax3.imshow(n_table, aspect='auto', cmap='Blues', vmin=0)
    plt.colorbar(im3, ax=ax3, label='Number of samples')
    ax3.set_title(f'Sample Count per Bin (min = {MIN_SAMPLES})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('15-min Slot')
    ax3.set_ylabel('Month')
    ax3.set_yticks(range(MONTHS))
    ax3.set_yticklabels(MONTH_NAMES)

    # --- Panel 4: Seasonal variation of max daily ratio ---
    ax4 = axes[1, 1]
    max_ratios   = [ratio_table[m].max() for m in range(MONTHS)]
    mean_ratios  = [ratio_table[m][ratio_table[m] > 0].mean()
                    if (ratio_table[m] > 0).any() else 0
                    for m in range(MONTHS)]
    x = np.arange(MONTHS)
    ax4.bar(x - 0.2, max_ratios,  width=0.35, label='Peak ratio (daily max)',
            color='orange', alpha=0.8, edgecolor='black')
    ax4.bar(x + 0.2, mean_ratios, width=0.35, label='Mean ratio (daytime)',
            color='steelblue', alpha=0.8, edgecolor='black')
    ax4.set_title('Seasonal Variation of Correction Ratios', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Ratio [kW / (W/m²)]')
    ax4.set_xticks(x)
    ax4.set_xticklabels(MONTH_NAMES)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add seasonal variation annotation
    valid_max = [r for r in max_ratios if r > 0]
    if valid_max:
        seasonal_factor = max(valid_max) / min(valid_max)
        ax4.text(0.97, 0.95, f'Seasonal factor: {seasonal_factor:.1f}×',
                 transform=ax4.transAxes, ha='right', va='top',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"  Analysis plot saved to: {OUT_PLOT}")


# =============================================================================
# VALIDATION: apply table to check PV estimate quality
# =============================================================================

def validate_correction_table(df, ratio_table):
    """
    Apply the correction table back to the training data and compare
    PV_Est vs PV_kW. Reports MAE and correlation as a sanity check.
    """
    mask = (df['GHI'] > IRR_THRESHOLD) & (df['PV_kW'] > 0)
    df_val = df[mask].copy()

    months = df_val['Month'].values - 1       # 0-indexed
    slots  = df_val['Slot_15min'].values       # 0-95
    ratio_per_row = ratio_table[months, slots]

    df_val['PV_Est'] = df_val['GHI'].values * ratio_per_row

    # Metrics
    pv_est = df_val['PV_Est'].values
    pv_act = df_val['PV_kW'].values
    mae    = np.mean(np.abs(pv_act - pv_est))
    rmse   = np.sqrt(np.mean((pv_act - pv_est) ** 2))
    corr   = np.corrcoef(pv_act, pv_est)[0, 1]
    mape   = np.mean(np.abs((pv_act - pv_est) / (pv_act + 1e-3))) * 100

    print(f"\n  Validation (training data self-check):")
    print(f"    Samples:  {len(df_val)}")
    print(f"    MAE:      {mae:.2f} kW")
    print(f"    RMSE:     {rmse:.2f} kW")
    print(f"    MAPE:     {mape:.1f}%")
    print(f"    R:        {corr:.3f}")
    print(f"    R²:       {corr**2:.3f}")
    print(f"  Note: Self-validation on training data gives upper-bound performance.")
    print(f"  Actual forecast skill will be lower due to cloud variability.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("BUILD PV IRRADIANCE-TO-POWER CORRECTION TABLE")
    print("Method: Lorenz (2011) / Bacher (2009)")
    print("=" * 70)

    # 1. Load data
    print("\n--- Step 1: Load PV data ---")
    df = load_pv_data(PV_FILE)

    # 2. Build correction table
    print("\n--- Step 2: Build correction table ---")
    ratio_table, n_table = build_correction_table(df)

    # 3. Validate (self-check on training data)
    print("\n--- Step 3: Validate ---")
    validate_correction_table(df, ratio_table)

    # 4. Save outputs
    print("\n--- Step 4: Save ---")
    save_outputs(ratio_table, n_table)

    # 5. Plot analysis
    print("\n--- Step 5: Plot analysis ---")
    plot_correction_analysis(df, ratio_table, n_table)

    print("\n" + "=" * 70)
    print("DONE — Correction table is ready for CNN_LSTM_Prediction.py")
    print(f"  NPZ: {OUT_NPZ}")
    print(f"  CSV: {OUT_CSV}")
    print("=" * 70)
