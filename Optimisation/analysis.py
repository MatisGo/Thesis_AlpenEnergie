"""
Analysis (Debug)
===============

Load the results CSV produced by `run_model.py`, print a quick overview of the
stored data, and plot a 1-day excerpt for debugging.

Usage:
  python analysis.py
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def load_results(path='results.csv'):
    """Load results and print a short summary of stored columns."""
    if not os.path.exists(path):
        print(f"ERROR: '{path}' not found.")
        print("  Run  python run_model.py  first.")
        return None

    df = pd.read_csv(path, parse_dates=['DateTime'])

    print(f"Loaded {len(df)} rows  ({df['DateTime'].iloc[0]} → {df['DateTime'].iloc[-1]})")
    print("\nColumns stored:")
    for col in df.columns:
        print(f"  - {col}  ({df[col].dtype})")

    print("\nSample data (first 5 rows):")
    print(df.head().to_string(index=False))

    return df


def plot_one_day(df, date=None):
    """Plot a 1-day excerpt for debugging (production, consumption, levels)."""
    if date is None:
        date = df['DateTime'].dt.date.iloc[0]

    day_df = df[df['DateTime'].dt.date == pd.to_datetime(date).date()]
    if day_df.empty:
        print(f"No data for {date}.")
        return

    fig, axes = plt.subplots(6, 1, figsize=(12, 19), sharex=True)

    axes[0].plot(day_df['DateTime'], day_df['Production_kW'], label='Total Production (kW)', color='tab:orange')
    axes[0].plot(day_df['DateTime'], day_df['Consumption_kW'], label='Consumption (kW)', color='tab:blue')
    axes[0].plot(day_df['DateTime'], day_df.get('Bidmi_Production_kW', pd.Series()),
                 label='Bidmi (MI) Prod', color='tab:green')
    axes[0].plot(day_df['DateTime'], day_df.get('Haselholz_Production_kW', pd.Series()),
                 label='Haselholz (MII) Prod', color='tab:red')
    axes[0].set_ylabel('Power (kW)')
    axes[0].set_title(f'Production vs Consumption  ({date})')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(day_df['DateTime'], day_df['Bidmi_mm'], label='Bidmi Level (mm)', color='tab:purple')
    axes[1].plot(day_df['DateTime'], day_df['Haselholz_mm'], label='Haselholz Level (mm)', color='tab:brown')
    axes[1].set_ylabel('Reservoir Level (mm)')
    axes[1].set_title('Reservoir Levels')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(day_df['DateTime'], day_df.get('Bidmi_Output_mm', pd.Series()),
                 label='Bidmi Turbine Output (mm)', color='tab:cyan')
    axes[2].plot(day_df['DateTime'], day_df.get('Haselholz_Output_mm', pd.Series()),
                 label='Haselholz Turbine Output (mm)', color='tab:olive')
    axes[2].set_ylabel('Turbine Water Output (mm / 5 min)')
    axes[2].set_title('Water Discharged Through Turbines per Timestep')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(day_df['DateTime'], day_df.get('Bidmi_Inflow_mm', pd.Series()),
                 label='Bidmi Inflow (mm)', color='tab:blue')
    axes[3].plot(day_df['DateTime'], day_df.get('Haselholz_Inflow_mm', pd.Series()),
                 label='Haselholz Inflow (mm)', color='tab:green')
    axes[3].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[3].set_ylabel('Water Inflow (mm / 5 min)')
    axes[3].set_title('Estimated Water Inflow per Timestep')
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    ax4b = axes[4].twinx()
    axes[4].plot(day_df['DateTime'], day_df.get('Spot_Price_CHF_MWh', pd.Series()),
                 label='Spot Price (CHF/MWh)', color='tab:pink')
    axes[4].plot(day_df['DateTime'], day_df.get('Network_Exchange_kW', pd.Series()),
                 label='Network Exchange (kW)', color='tab:gray')
    axes[4].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[4].set_ylabel('Spot Price (CHF/MWh) / Exchange (kW)')
    axes[4].set_title('Spot Price & Network Exchange')
    axes[4].legend(fontsize=9, loc='upper left')
    axes[4].grid(True, alpha=0.3)

    ax4b.plot(day_df['DateTime'], day_df.get('Energy_Trading_CHF', pd.Series()),
              label='Energy Trading (CHF)', color='tab:red', linestyle='--', alpha=0.7)
    ax4b.axhline(0, color='tab:red', linewidth=0.5, linestyle=':')
    ax4b.set_ylabel('Energy Trading (CHF / 5 min)')
    ax4b.legend(fontsize=9, loc='upper right')

    # Daily cumulative cost: reset to 0 at start of the day
    daily_trading = day_df.get('Energy_Trading_CHF', pd.Series(dtype=float))
    daily_cumcost = daily_trading.cumsum() - daily_trading.cumsum().iloc[0]
    daily_total = daily_trading.sum()

    axes[5].plot(day_df['DateTime'], daily_cumcost,
                 label='Cumulative Energy Cost (CHF)', color='tab:red')
    axes[5].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[5].set_ylabel('Cumulative Cost (CHF)')
    axes[5].set_title(f'Total Energy Cost  —  Day total: {daily_total:+.2f} CHF')
    axes[5].legend(fontsize=9)
    axes[5].grid(True, alpha=0.3)

    axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[5].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    plt.setp(axes[5].xaxis.get_majorticklabels(), rotation=45, ha='right')

    print(f"\nTotal energy cost for {date}: {daily_total:+.2f} CHF")
    print(f"  (positive = net buyer, negative = net seller)")

    plt.tight_layout()
    plt.savefig('debug_one_day.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: debug_one_day.png")


def plot_optimised_day(df, date=None):
    """Compare actual vs optimised production, reservoir levels, and cost for one day.

    Requires optimised_results.csv produced by optimise.py.
    """
    if not os.path.exists('optimised_results.csv'):
        print("ERROR: 'optimised_results.csv' not found.")
        print("  Run  python optimise.py  first.")
        return

    opt_df = pd.read_csv('optimised_results.csv', parse_dates=['DateTime'])

    if date is None:
        date = df['DateTime'].dt.date.iloc[0]

    target = pd.to_datetime(date).date()
    day_a = df[df['DateTime'].dt.date == target]
    day_o = opt_df[opt_df['DateTime'].dt.date == target]

    if day_a.empty or day_o.empty:
        print(f"No data for {date}.")
        return

    _, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Production ---
    axes[0].plot(day_a['DateTime'], day_a['Production_kW'],
                 label='Actual (kW)', color='tab:orange')
    axes[0].plot(day_o['DateTime'], day_o['Opt_Production_kW'],
                 label='Optimised (kW)', color='tab:blue', linestyle='--')
    axes[0].plot(day_a['DateTime'], day_a['Consumption_kW'],
                 label='Demand (kW)', color='tab:gray', linewidth=0.8, linestyle=':')
    axes[0].set_ylabel('Power (kW)')
    axes[0].set_title(f'Actual vs Optimised Production  ({date})')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Reservoir levels ---
    axes[1].plot(day_a['DateTime'], day_a['Bidmi_mm'],
                 label='Bidmi actual (mm)', color='tab:purple')
    axes[1].plot(day_o['DateTime'], day_o['Opt_Bidmi_mm'],
                 label='Bidmi optimised (mm)', color='tab:purple', linestyle='--')
    axes[1].plot(day_a['DateTime'], day_a['Haselholz_mm'],
                 label='Haselholz actual (mm)', color='tab:brown')
    axes[1].plot(day_o['DateTime'], day_o['Opt_Haselholz_mm'],
                 label='Haselholz optimised (mm)', color='tab:brown', linestyle='--')
    axes[1].set_ylabel('Reservoir Level (mm)')
    axes[1].set_title('Actual vs Optimised Reservoir Levels')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # --- Cumulative energy cost ---
    act_trade  = day_a['Energy_Trading_CHF']
    opt_trade  = day_o['Opt_Energy_Trading_CHF']
    act_cum    = act_trade.cumsum() - act_trade.cumsum().iloc[0]
    opt_cum    = opt_trade.cumsum() - opt_trade.cumsum().iloc[0]
    act_total  = act_trade.sum()
    opt_total  = opt_trade.sum()
    # opt_gain > 0 means optimizer is better (lower cost / higher revenue)
    # opt_gain < 0 means actual outperformed optimizer on this day
    opt_gain   = act_total - opt_total

    if opt_gain >= 0:
        gain_label = f'Opt saves {opt_gain:+.2f} CHF'
    else:
        gain_label = f'Actual better by {-opt_gain:.2f} CHF'

    axes[2].plot(day_a['DateTime'], act_cum,
                 label=f'Actual ({act_total:+.2f} CHF)', color='tab:red')
    axes[2].plot(day_o['DateTime'], opt_cum,
                 label=f'Optimised ({opt_total:+.2f} CHF)', color='tab:green', linestyle='--')
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].set_ylabel('Cumulative Cost (CHF)  [negative = revenue]')
    axes[2].set_title(f'Cumulative Energy Cost  —  {gain_label}')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[2].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

    print(f"\nOptimisation comparison for {date}:")
    print(f"  Actual cost     : {act_total:+.2f} CHF  (negative = net revenue from selling)")
    print(f"  Optimised cost  : {opt_total:+.2f} CHF  (negative = net revenue from selling)")
    print(f"  Optimizer gain  : {opt_gain:+.2f} CHF")
    if opt_gain >= 0:
        print(f"  -> Optimizer saves {opt_gain:.2f} CHF vs actual")
    else:
        print(f"  -> Actual outperformed optimizer by {-opt_gain:.2f} CHF"
              f" (terminal constraint limited production)")

    plt.tight_layout()
    plt.savefig('optimised_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: optimised_vs_actual.png")


if __name__ == '__main__':
    df = load_results('results.csv')
    if df is not None:
        plot_one_day(df)
        plot_optimised_day(df)
