"""
Daily Dashboard
===============

Plots a five-panel overview for a selected day:

  1. Bidmi reservoir level       (actual vs optimised + seasonal target + physical bounds)
  2. Haselholz reservoir level   (actual vs optimised + seasonal target + physical bounds)
  3. M2 (Bidmi) production       (reference vs optimised)
  4. M1 (Haselholz) production   (reference vs optimised)
  5. Spot price (left axis) + cumulative energy cost reference vs optimised (right axis)

Data source: optimised_results.csv  (produced by python optimise.py)
  - Reference production : columns Ref_M2_kW / Ref_M1_kW  (Excel cols L / M)
  - Optimised production : columns Opt_M2_kW / Opt_M1_kW

Usage:
  python analysis.py              # first day in dataset
  python analysis.py 2025-06-15   # specific date  (YYYY-MM-DD)
"""

import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OPTIMISED_FILE = 'optimised_results.csv'

# Physical bounds (must match optimise.py)
BIDMI_MIN        = 1000.0
BIDMI_MAX        = 2200.0
HASELHOLZ_MIN    =  600.0
HASELHOLZ_MAX    = 2800.0
M2_MAX_KW        = 1700.0
M1_MAX_KW        = 1156.0

# Colours
C_REF    = '#2166ac'    # blue   — reference (historical)
C_OPT    = '#d6604d'    # red-orange — optimised
C_TARGET = '#888888'    # grey   — seasonal target
C_PRICE  = '#762a83'    # purple — spot price
C_BOUND  = '#b2182b'    # dark red — physical bounds


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_day(date_str=None):
    """Return (date, df) for the requested day from optimised_results.csv."""

    if not os.path.exists(OPTIMISED_FILE):
        print(f"ERROR: '{OPTIMISED_FILE}' not found.")
        print("  Run  python optimise.py  first.")
        sys.exit(1)

    df = pd.read_csv(OPTIMISED_FILE, parse_dates=['DateTime'])

    if date_str is None:
        date = df['DateTime'].dt.date.iloc[0]
        print(f"No date specified — plotting first day: {date}")
    else:
        date = pd.to_datetime(date_str).date()

    day = df[df['DateTime'].dt.date == date].copy().reset_index(drop=True)

    if day.empty:
        print(f"ERROR: no data found for {date}.")
        sys.exit(1)

    has_opt = 'Opt_M2_kW' in day.columns and day['Opt_M2_kW'].notna().any()

    return date, day, has_opt


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_day(date, day, has_opt):
    """Create and save the five-panel dashboard."""

    t = day['DateTime']

    fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)
    fig.suptitle(f'Daily Dashboard  —  {date}', fontsize=15, fontweight='bold', y=0.995)

    # ------------------------------------------------------------------ #
    # Panel 1 — Bidmi reservoir level
    # ------------------------------------------------------------------ #
    ax = axes[0]

    ax.plot(t, day['Bidmi_mm'],
            color=C_REF, linewidth=1.8, label='Actual level')

    if has_opt:
        ax.plot(t, day['Opt_Bidmi_mm'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised level')

        if 'Opt_Target_Bidmi_mm' in day.columns:
            tgt = day['Opt_Target_Bidmi_mm'].iloc[0]
            ax.axhline(tgt, color=C_TARGET, linewidth=1.2, linestyle=':',
                       label=f'Seasonal target  {tgt:.0f} mm')
            ax.axhspan(tgt - 100, tgt + 100, alpha=0.07, color=C_TARGET,
                       label='±100 mm free zone')

    ax.axhline(BIDMI_MAX, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Max  {BIDMI_MAX:.0f} mm')
    ax.axhline(BIDMI_MIN, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Min  {BIDMI_MIN:.0f} mm')

    ax.set_ylabel('Level (mm)')
    ax.set_title('Bidmi — Reservoir Level')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------ #
    # Panel 2 — Haselholz reservoir level
    # ------------------------------------------------------------------ #
    ax = axes[1]

    ax.plot(t, day['Haselholz_mm'],
            color=C_REF, linewidth=1.8, label='Actual level')

    if has_opt:
        ax.plot(t, day['Opt_Haselholz_mm'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised level')

        if 'Opt_Target_Haselholz_mm' in day.columns:
            tgt = day['Opt_Target_Haselholz_mm'].iloc[0]
            ax.axhline(tgt, color=C_TARGET, linewidth=1.2, linestyle=':',
                       label=f'Seasonal target  {tgt:.0f} mm')
            ax.axhspan(tgt - 100, tgt + 100, alpha=0.07, color=C_TARGET,
                       label='±100 mm free zone')

    ax.axhline(HASELHOLZ_MAX, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Max  {HASELHOLZ_MAX:.0f} mm')
    ax.axhline(HASELHOLZ_MIN, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Min  {HASELHOLZ_MIN:.0f} mm')

    ax.set_ylabel('Level (mm)')
    ax.set_title('Haselholz — Reservoir Level')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------ #
    # Panel 3 — M2 (Bidmi) turbine production
    # ------------------------------------------------------------------ #
    ax = axes[2]

    ax.fill_between(t, day['Ref_M2_kW'], alpha=0.15, color=C_REF)
    ax.plot(t, day['Ref_M2_kW'],
            color=C_REF, linewidth=1.8, label='Reference')

    if has_opt:
        ax.plot(t, day['Opt_M2_kW'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised')

    ax.axhline(M2_MAX_KW, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Rated max  {M2_MAX_KW:.0f} kW')

    ax.plot(t, day['Consumption_kW'],
            color='#4d4d4d', linewidth=1.0, linestyle=':', alpha=0.7, label='Consumption')

    ax.set_ylabel('Power (kW)')
    ax.set_title('M2 (Bidmi) — Turbine Production')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------ #
    # Panel 4 — M1 (Haselholz) turbine production
    # ------------------------------------------------------------------ #
    ax = axes[3]

    ax.fill_between(t, day['Ref_M1_kW'], alpha=0.15, color=C_REF)
    ax.plot(t, day['Ref_M1_kW'],
            color=C_REF, linewidth=1.8, label='Reference')

    if has_opt:
        ax.plot(t, day['Opt_M1_kW'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised')

    ax.axhline(M1_MAX_KW, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Rated max  {M1_MAX_KW:.0f} kW')

    ax.plot(t, day['Consumption_kW'],
            color='#4d4d4d', linewidth=1.0, linestyle=':', alpha=0.7, label='Consumption')

    ax.set_ylabel('Power (kW)')
    ax.set_title('M1 (Haselholz) — Turbine Production')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------ #
    # Panel 5 — Spot price (left) + cumulative energy cost (right)
    # ------------------------------------------------------------------ #
    ax  = axes[4]
    ax2 = ax.twinx()

    ax.plot(t, day['Spot_Price_CHF_MWh'],
            color=C_PRICE, linewidth=1.5, alpha=0.9, label='Spot price')
    ax.set_ylabel('Spot Price (CHF / MWh)', color=C_PRICE)
    ax.tick_params(axis='y', labelcolor=C_PRICE)
    ax.set_ylim(bottom=0)

    # Cumulative cost — reset to 0 at day start
    ref_trade = day['Ref_Energy_Trading_CHF']
    ref_cum   = ref_trade.cumsum() - ref_trade.cumsum().iloc[0]
    ref_total = ref_trade.sum()

    ax2.plot(t, ref_cum,
             color=C_REF, linewidth=1.8,
             label=f'Reference  {ref_total:+.2f} CHF')

    if has_opt:
        opt_trade = day['Opt_Energy_Trading_CHF']
        opt_cum   = opt_trade.cumsum() - opt_trade.cumsum().iloc[0]
        opt_total = opt_trade.sum()

        ax2.plot(t, opt_cum,
                 color=C_OPT, linewidth=1.8, linestyle='--',
                 label=f'Optimised  {opt_total:+.2f} CHF')

        gain = opt_total - ref_total
        gain_str = (f'Opt earns {gain:+.2f} CHF more'
                    if gain >= 0 else f'Reference earns {-gain:.2f} CHF more')
        title = f'Spot Price & Cumulative Cost  [{gain_str}]'
    else:
        title = 'Spot Price & Cumulative Cost'

    ax2.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax2.set_ylabel('Cumulative Profit (CHF)   [positive = net seller]')

    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

    # ------------------------------------------------------------------ #
    # X-axis ticks — every 2 hours
    # ------------------------------------------------------------------ #
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), 'Dashboard Results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'dashboard_{date}.png')
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.show()

    # ------------------------------------------------------------------ #
    # Console summary
    # ------------------------------------------------------------------ #
    print(f"\n=== {date} ===")
    print(f"  Bidmi    level : {day['Bidmi_mm'].iloc[0]:.0f} -> {day['Bidmi_mm'].iloc[-1]:.0f} mm"
          f"  (range {day['Bidmi_mm'].min():.0f} - {day['Bidmi_mm'].max():.0f})")
    print(f"  Haselholz level: {day['Haselholz_mm'].iloc[0]:.0f} -> {day['Haselholz_mm'].iloc[-1]:.0f} mm"
          f"  (range {day['Haselholz_mm'].min():.0f} - {day['Haselholz_mm'].max():.0f})")
    print(f"  Reference prod avg : {(day['Ref_M2_kW'] + day['Ref_M1_kW']).mean():.0f} kW"
          f"  (M2 {day['Ref_M2_kW'].mean():.0f} + M1 {day['Ref_M1_kW'].mean():.0f})")
    print(f"  Reference profit: {ref_total:+.2f} CHF  (positive = net seller)")

    if has_opt:
        print(f"  Optimised prod avg : {day['Opt_Production_kW'].mean():.0f} kW"
              f"  (M2 {day['Opt_M2_kW'].mean():.0f} + M1 {day['Opt_M1_kW'].mean():.0f})")
        print(f"  Optimised profit: {opt_total:+.2f} CHF")
        print(f"  --> {gain_str}")

    print(f"\nSaved: {out_file}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    date, day, has_opt = load_day(date_str)
    plot_day(date, day, has_opt)
