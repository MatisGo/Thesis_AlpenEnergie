"""
Daily Dashboard
===============

Plots a five-panel overview for a selected day:

  1. Bidmi reservoir level       (actual vs optimised + seasonal target + physical bounds)
  2. Haselholz reservoir level   (actual vs optimised + seasonal target + physical bounds)
  3. Bidmi turbine production    (actual vs optimised + consumption reference)
  4. Haselholz turbine production(actual vs optimised + consumption reference)
  5. Spot price (left axis) + cumulative energy cost actual vs optimised (right axis)

Usage:
  python analysis.py              # first day in dataset
  python analysis.py 2025-06-15   # specific date  (YYYY-MM-DD)
"""

import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILE   = 'results.csv'
OPTIMISED_FILE = 'optimised_results.csv'

# Physical bounds (must match optimise.py)
BIDMI_MIN    = 1000.0
BIDMI_MAX    = 2200.0
HASELHOLZ_MIN =  600.0
HASELHOLZ_MAX = 2800.0
BIDMI_MAX_KW    = 1700.0
HASELHOLZ_MAX_KW = 1156.0

# Colours
C_ACTUAL = '#2166ac'    # blue
C_OPT    = '#d6604d'    # red-orange
C_TARGET = '#888888'    # grey
C_PRICE  = '#762a83'    # purple
C_BOUND  = '#b2182b'    # dark red


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_day(date_str=None):
    """Return (date, actual_df, opt_df_or_None) for the requested day."""

    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: '{RESULTS_FILE}' not found. Run  python run_model.py  first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_FILE, parse_dates=['DateTime'])

    if date_str is None:
        date = df['DateTime'].dt.date.iloc[0]
        print(f"No date specified — plotting first day: {date}")
    else:
        date = pd.to_datetime(date_str).date()

    actual = df[df['DateTime'].dt.date == date].copy().reset_index(drop=True)
    if actual.empty:
        print(f"ERROR: no data found for {date}.")
        sys.exit(1)

    opt = None
    if os.path.exists(OPTIMISED_FILE):
        opt_df = pd.read_csv(OPTIMISED_FILE, parse_dates=['DateTime'])
        opt_day = opt_df[opt_df['DateTime'].dt.date == date].copy().reset_index(drop=True)
        if not opt_day.empty:
            opt = opt_day
        else:
            print(f"Note: '{OPTIMISED_FILE}' exists but has no data for {date}.")
    else:
        print(f"Note: '{OPTIMISED_FILE}' not found — showing actual data only.")

    return date, actual, opt


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_day(date, actual, opt):
    """Create and save the five-panel dashboard."""

    has_opt = opt is not None
    t_a = actual['DateTime']
    t_o = opt['DateTime'] if has_opt else None

    fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)
    fig.suptitle(f'Daily Dashboard  —  {date}', fontsize=15, fontweight='bold', y=0.995)

    # ------------------------------------------------------------------ #
    # Panel 1 — Bidmi reservoir level
    # ------------------------------------------------------------------ #
    ax = axes[0]

    ax.plot(t_a, actual['Bidmi_mm'],
            color=C_ACTUAL, linewidth=1.8, label='Actual level')

    if has_opt:
        ax.plot(t_o, opt['Opt_Bidmi_mm'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised level')

        if 'Opt_Target_Bidmi_mm' in opt.columns:
            tgt = opt['Opt_Target_Bidmi_mm'].iloc[0]
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

    ax.plot(t_a, actual['Haselholz_mm'],
            color=C_ACTUAL, linewidth=1.8, label='Actual level')

    if has_opt:
        ax.plot(t_o, opt['Opt_Haselholz_mm'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised level')

        if 'Opt_Target_Haselholz_mm' in opt.columns:
            tgt = opt['Opt_Target_Haselholz_mm'].iloc[0]
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
    # Panel 3 — Bidmi turbine production
    # ------------------------------------------------------------------ #
    ax = axes[2]

    ax.fill_between(t_a, actual['Bidmi_Production_kW'],
                    alpha=0.15, color=C_ACTUAL)
    ax.plot(t_a, actual['Bidmi_Production_kW'],
            color=C_ACTUAL, linewidth=1.8, label='Actual')

    if has_opt:
        ax.plot(t_o, opt['Opt_Bidmi_Production_kW'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised')

    ax.axhline(BIDMI_MAX_KW, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Rated max  {BIDMI_MAX_KW:.0f} kW')

    # Consumption reference (what the plant needs to cover)
    ax.plot(t_a, actual['Consumption_kW'],
            color='#4d4d4d', linewidth=1.0, linestyle=':', alpha=0.7, label='Consumption')

    ax.set_ylabel('Power (kW)')
    ax.set_title('Bidmi — Turbine Production')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------ #
    # Panel 4 — Haselholz turbine production
    # ------------------------------------------------------------------ #
    ax = axes[3]

    ax.fill_between(t_a, actual['Haselholz_Production_kW'],
                    alpha=0.15, color=C_ACTUAL)
    ax.plot(t_a, actual['Haselholz_Production_kW'],
            color=C_ACTUAL, linewidth=1.8, label='Actual')

    if has_opt:
        ax.plot(t_o, opt['Opt_Haselholz_Production_kW'],
                color=C_OPT, linewidth=1.8, linestyle='--', label='Optimised')

    ax.axhline(HASELHOLZ_MAX_KW, color=C_BOUND, linewidth=0.9, linestyle='--', alpha=0.6,
               label=f'Rated max  {HASELHOLZ_MAX_KW:.0f} kW')

    ax.plot(t_a, actual['Consumption_kW'],
            color='#4d4d4d', linewidth=1.0, linestyle=':', alpha=0.7, label='Consumption')

    ax.set_ylabel('Power (kW)')
    ax.set_title('Haselholz — Turbine Production')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------ #
    # Panel 5 — Spot price (left) + cumulative energy cost (right)
    # ------------------------------------------------------------------ #
    ax   = axes[4]
    ax2  = ax.twinx()

    # Spot price
    ax.plot(t_a, actual['Spot_Price_CHF_MWh'],
            color=C_PRICE, linewidth=1.5, alpha=0.9, label='Spot price')
    ax.set_ylabel('Spot Price (CHF / MWh)', color=C_PRICE)
    ax.tick_params(axis='y', labelcolor=C_PRICE)
    ax.set_ylim(bottom=0)

    # Cumulative cost — reset to 0 at day start
    act_trade = actual['Energy_Trading_CHF']
    act_cum   = act_trade.cumsum() - act_trade.cumsum().iloc[0]
    act_total = act_trade.sum()

    ax2.plot(t_a, act_cum,
             color=C_ACTUAL, linewidth=1.8,
             label=f'Actual  {act_total:+.2f} CHF')

    if has_opt:
        opt_trade = opt['Opt_Energy_Trading_CHF']
        opt_cum   = opt_trade.cumsum() - opt_trade.cumsum().iloc[0]
        opt_total = opt_trade.sum()

        ax2.plot(t_o, opt_cum,
                 color=C_OPT, linewidth=1.8, linestyle='--',
                 label=f'Optimised  {opt_total:+.2f} CHF')

        gain = act_total - opt_total
        gain_str = (f'Opt saves {gain:+.2f} CHF'
                    if gain >= 0 else f'Actual better by {-gain:.2f} CHF')
        title = f'Spot Price & Cumulative Cost  [{gain_str}]'
    else:
        title = 'Spot Price & Cumulative Cost'

    ax2.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax2.set_ylabel('Cumulative Cost (CHF)   [negative = revenue]')

    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Combined legend for dual-axis panel
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

    out_file = f'dashboard_{date}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.show()

    # ------------------------------------------------------------------ #
    # Console summary
    # ------------------------------------------------------------------ #
    print(f"\n=== {date} ===")
    print(f"  Bidmi    level : {actual['Bidmi_mm'].iloc[0]:.0f} -> {actual['Bidmi_mm'].iloc[-1]:.0f} mm"
          f"  (range {actual['Bidmi_mm'].min():.0f} – {actual['Bidmi_mm'].max():.0f})")
    print(f"  Haselholz level: {actual['Haselholz_mm'].iloc[0]:.0f} -> {actual['Haselholz_mm'].iloc[-1]:.0f} mm"
          f"  (range {actual['Haselholz_mm'].min():.0f} – {actual['Haselholz_mm'].max():.0f})")
    print(f"  Actual production avg : {actual['Production_kW'].mean():.0f} kW"
          f"  (Bidmi {actual['Bidmi_Production_kW'].mean():.0f}"
          f" + Haselholz {actual['Haselholz_Production_kW'].mean():.0f})")
    print(f"  Actual energy cost    : {act_total:+.2f} CHF  (negative = net revenue)")

    if has_opt:
        print(f"  Optimised prod avg    : {opt['Opt_Production_kW'].mean():.0f} kW"
              f"  (Bidmi {opt['Opt_Bidmi_Production_kW'].mean():.0f}"
              f" + Haselholz {opt['Opt_Haselholz_Production_kW'].mean():.0f})")
        print(f"  Optimised energy cost : {opt_total:+.2f} CHF")
        print(f"  --> {gain_str}")

    print(f"\nSaved: {out_file}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    date, actual, opt = load_day(date_str)
    plot_day(date, actual, opt)
