"""
Analysis
========
Load optimisation results and create diagnostic plots.

Run this script after run_model.py has produced results.csv.

Plots produced:
  1. plot_reservoirs.png   → Bidmi and Haselholz water levels over time
  2. plot_production.png   → Turbine discharge + production vs demand
  3. plot_market.png       → Electricity bought and sold on markets

Usage:
  python analysis.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ===========================================================================
# LOAD RESULTS
# ===========================================================================

def load_results(path='results.csv'):
    """Load the optimisation results CSV produced by run_model.py."""
    if not os.path.exists(path):
        print(f"ERROR: '{path}' not found.")
        print("  Run  python run_model.py  first.")
        return None

    df = pd.read_csv(path, parse_dates=['DateTime'])
    print(f"Loaded {len(df)} timesteps  "
          f"({df['DateTime'].iloc[0].date()} → {df['DateTime'].iloc[-1].date()})")
    return df


# ===========================================================================
# PLOT 1: Reservoir Levels
# ===========================================================================

def plot_reservoirs(df):
    """
    Plot the optimised water level for both reservoirs over time.
    Shows whether the reservoirs are being drawn down or refilled.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Bidmi reservoir (fed by M2 turbine)
    axes[0].plot(df['DateTime'], df['R_Bidmi_mm'],
                 color='royalblue', linewidth=1.5, label='Bidmi')
    axes[0].set_ylabel('Water Level (mm)', fontsize=11)
    axes[0].set_title('Bidmi Reservoir  (Turbine M2)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Haselholz reservoir (fed by M1 turbine)
    axes[1].plot(df['DateTime'], df['R_Haselholz_mm'],
                 color='steelblue', linewidth=1.5, label='Haselholz')
    axes[1].set_ylabel('Water Level (mm)', fontsize=11)
    axes[1].set_title('Haselholz Reservoir  (Turbine M1)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    # Format x-axis dates
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Reservoir Water Levels — Optimised', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_reservoirs.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_reservoirs.png")


# ===========================================================================
# PLOT 2: Turbine Discharge and Production
# ===========================================================================

def plot_production(df):
    """
    Plot turbine water discharge and electricity production vs demand.
    Two panels: top = water discharged, bottom = electricity generated.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Turbine water discharge  [mm per 5-min step]
    axes[0].plot(df['DateTime'], df['Q_M1_mm'],
                 color='steelblue', linewidth=1.2, label='Q_M1  (Haselholz)')
    axes[0].plot(df['DateTime'], df['Q_M2_mm'],
                 color='royalblue', linewidth=1.2, label='Q_M2  (Bidmi)')
    axes[0].set_ylabel('Discharge  (mm / 5-min)', fontsize=11)
    axes[0].set_title('Turbine Water Discharge', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Electricity production vs demand  [kWh per 5-min step]
    axes[1].plot(df['DateTime'], df['P_total_kWh'],
                 color='darkorange', linewidth=1.5, label='Total Production')
    axes[1].plot(df['DateTime'], df['Demand_kWh'],
                 color='black', linewidth=1.2, linestyle='--', label='Demand')
    axes[1].set_ylabel('Energy  (kWh / 5-min)', fontsize=11)
    axes[1].set_title('Production vs Demand', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Format x-axis
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Turbine Operation — Optimised', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_production.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_production.png")


# ===========================================================================
# PLOT 3: Market Decisions
# ===========================================================================

def plot_market(df):
    """
    Plot electricity market activity.
    SpotTrade > 0 = selling (revenue, shown above zero line in green).
    SpotTrade < 0 = buying  (cost,    shown below zero line in red).
    IntradayBuy always shown above zero line in orange.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Separate sell (>0) and buy (<0) from the single SpotTrade column
    spot_sell = df['SpotTrade_kWh'].clip(lower=0)   # keep only positive values
    spot_buy  = df['SpotTrade_kWh'].clip(upper=0)   # keep only negative values

    ax.fill_between(df['DateTime'], spot_sell,
                    alpha=0.5, color='green',  label='Spot Sell  (+revenue)')
    ax.fill_between(df['DateTime'], spot_buy,
                    alpha=0.5, color='red',    label='Spot Buy   (−cost)')
    ax.fill_between(df['DateTime'], df['IntradayBuy_kWh'],
                    alpha=0.4, color='orange', label='Intraday Buy  (+cost)')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Energy  (kWh / 5-min)', fontsize=11)
    ax.set_title('Electricity Market Decisions — Optimised',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('plot_market.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_market.png")


# ===========================================================================
# KPI SUMMARY
# ===========================================================================

def print_kpis(df):
    """Print key performance indicators from the optimisation results."""

    print("\n" + "=" * 50)
    print("  OPTIMISATION KPIs")
    print("=" * 50)

    period_start = df['DateTime'].iloc[0].date()
    period_end   = df['DateTime'].iloc[-1].date()
    n_steps      = len(df)
    n_days       = (df['DateTime'].iloc[-1] - df['DateTime'].iloc[0]).days + 1

    print(f"  Period:              {period_start}  →  {period_end}  ({n_days} days)")
    print(f"  Timesteps:           {n_steps}")
    print()
    print(f"  Total production:    {df['P_total_kWh'].sum():>10.1f}  kWh")
    print(f"    → M1 (Haselholz):  {df['P_M1_kWh'].sum():>10.1f}  kWh")
    print(f"    → M2 (Bidmi):      {df['P_M2_kWh'].sum():>10.1f}  kWh")
    print()
    print(f"  Total demand:        {df['Demand_kWh'].sum():>10.1f}  kWh")
    print(f"  Spot sold  (+):      {df['SpotTrade_kWh'].clip(lower=0).sum():>10.1f}  kWh")
    print(f"  Spot bought (-):     {df['SpotTrade_kWh'].clip(upper=0).abs().sum():>10.1f}  kWh")
    print(f"  Intraday buy:        {df['IntradayBuy_kWh'].sum():>10.1f}  kWh")
    print()

    # Financial summary
    # SpotTrade > 0 = revenue, SpotTrade < 0 = cost
    revenue  = (df['SpotTrade_kWh'].clip(lower=0)       * df['Price_spot']).sum()
    buy_cost = (df['SpotTrade_kWh'].clip(upper=0).abs() * df['Price_spot']).sum()
    net_cost = buy_cost - revenue

    print(f"  Revenue from selling:{revenue:>10.2f}  €")
    print(f"  Cost of buying:      {buy_cost:>10.2f}  €")
    print(f"  Net cost:            {net_cost:>10.2f}  €")
    print("=" * 50)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':

    df = load_results()

    if df is not None:
        print_kpis(df)
        plot_reservoirs(df)
        plot_production(df)
        plot_market(df)
