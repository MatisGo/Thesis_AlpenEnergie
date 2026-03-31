"""
run_optimiser.py
================
Entry point for all hydro scheduling modes.

Usage
-----
  python run_optimiser.py               # all days
  python run_optimiser.py 2025-06-15    # single day

Change MODE below to switch between models.
"""

import os
import sys
import pandas as pd

from hydro_constants import (
    DATA_FILENAME, TIMESTEP_HOURS,
    load_data, compute_seasonal_targets, compute_intraday_targets,
)

# ===========================================================================
#  SELECT MODE HERE
# ===========================================================================
#
#   'FORECAST'         Rule-based 3-state equalization (follows CNN/LSTM forecast)
#   'PRICE_ARBITRAGE'  Pure LP profit maximisation
#   'WATER_VALUE'      LP that follows forecast + optimises production timing
#   'WATER_LEVEL'      Both turbines off (reservoir recovery)
#
MODE = 'WATER_VALUE'
# ===========================================================================

OUTPUT_FILENAME = {
    'FORECAST':        'Forecast_results.xlsx',
    'PRICE_ARBITRAGE': 'Price_Arbitrage_results.xlsx',
    'WATER_VALUE':     'Water_Value_results.xlsx',
    'WATER_LEVEL':     'Water_Level_results.xlsx',
}.get(MODE, 'results.xlsx')

# Import the correct mode module
if MODE == 'FORECAST':
    from mode_forecast import dispatch_day
elif MODE == 'PRICE_ARBITRAGE':
    from mode_price_arbitrage import dispatch_day, get_solver
elif MODE == 'WATER_VALUE':
    from mode_water_value import dispatch_day
    from mode_price_arbitrage import get_solver
elif MODE == 'WATER_LEVEL':
    from mode_water_level import dispatch_day
else:
    raise ValueError(f"Unknown MODE: '{MODE}'")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
    """Iterate over all days, chain reservoir levels, return full result DataFrame."""

    print("Computing seasonal targets...")
    season_targets = compute_seasonal_targets(df)

    # LP modes need intra-day floor targets and a solver
    needs_lp     = MODE in ('PRICE_ARBITRAGE', 'WATER_VALUE')
    solver       = get_solver() if needs_lp else None
    intraday_tgt = compute_intraday_targets(df) if needs_lp else {}

    days = sorted(df['DateTime'].dt.date.unique())
    if target_date:
        import datetime
        days = [d for d in days if str(d) == target_date]
        if not days:
            print(f"ERROR: date '{target_date}' not found in data.")
            sys.exit(1)

    results  = []
    prev_lb  = None
    prev_lh  = None

    for i, day in enumerate(days, 1):
        day_df = df[df['DateTime'].dt.date == day].copy()

        t_lb, t_lh = season_targets.get(day, (None, None))
        f_lb, f_lh = intraday_tgt.get(day, (None, None))

        if t_lb is None: t_lb = float(day_df['Bidmi_mm'].iloc[0])
        if t_lh is None: t_lh = float(day_df['Haselholz_mm'].iloc[0])

        lb0 = prev_lb if prev_lb is not None else float(day_df['Bidmi_mm'].iloc[0])
        lh0 = prev_lh if prev_lh is not None else float(day_df['Haselholz_mm'].iloc[0])

        day_df = dispatch_day(
            day_df, lb0, lh0, t_lb, t_lh,
            solver=solver, floor_lb=f_lb, floor_lh=f_lh,
        )

        prev_lb = getattr(day_df, '_opt_lb_end', None)
        prev_lh = getattr(day_df, '_opt_lh_end', None)

        # Console progress
        opt_profit = day_df['Opt_Energy_Trading_CHF'].sum()
        ref_profit = day_df['Ref_Energy_Trading_CHF'].sum()
        dev_b      = getattr(day_df, '_dev_b', float('nan'))
        dev_h      = getattr(day_df, '_dev_h', float('nan'))
        gain       = opt_profit - ref_profit
        spill      = day_df['Opt_Spill_Bidmi_kWh'].sum() + day_df['Opt_Spill_Haselholz_kWh'].sum()
        spill_str  = f"  SPILL {spill:.0f} kWh" if spill > 0.1 else ""

        print(f"[{i:3}/{len(days)}] {day}  "
              f"tgt B={t_lb:.0f} H={t_lh:.0f}  "
              f"ref {ref_profit:+.2f} CHF  opt {opt_profit:+.2f} CHF  "
              f"gain {gain:+.2f} CHF  "
              f"dLB={dev_b:+.1f}mm dLH={dev_h:+.1f}mm"
              f"{spill_str}")

        results.append(day_df)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_output(opt_df: pd.DataFrame, output_path: str):
    """Save Results + Summary sheets to Excel."""
    opt_df['Opt_Total_Energy_Cost_CHF'] = opt_df['Opt_Energy_Trading_CHF'].cumsum()

    total_ref  = opt_df['Ref_Energy_Trading_CHF'].sum()
    total_opt  = opt_df['Opt_Energy_Trading_CHF'].sum()
    total_gain = total_opt - total_ref

    daily_ref        = opt_df.groupby(opt_df['DateTime'].dt.date)['Ref_Energy_Trading_CHF'].sum()
    daily_opt        = opt_df.groupby(opt_df['DateTime'].dt.date)['Opt_Energy_Trading_CHF'].sum()
    days_opt_better  = (daily_opt > daily_ref).sum()
    days_ref_better  = (daily_opt < daily_ref).sum()

    ref_energy_kwh = ((opt_df['Ref_M2_kW'] + opt_df['Ref_M1_kW']) * TIMESTEP_HOURS).sum()
    opt_energy_kwh = opt_df['Opt_Production_kW'].sum() * TIMESTEP_HOURS
    energy_diff    = opt_energy_kwh - ref_energy_kwh

    total_spill_b   = opt_df['Opt_Spill_Bidmi_kWh'].sum()
    total_spill_h   = opt_df['Opt_Spill_Haselholz_kWh'].sum()
    total_spill     = total_spill_b + total_spill_h

    # Forecast drift (FORECAST and WATER_VALUE modes)
    drift_rows = []
    if 'Forecast_Drift_kW' in opt_df.columns:
        drift     = opt_df['Forecast_Drift_kW']
        drift_kwh = (drift * TIMESTEP_HOURS).sum()
        mae_kw    = drift.abs().mean()
        n         = len(drift)
        h_below   = (drift < -1.0).sum() * TIMESTEP_HOURS
        h_above   = (drift >  1.0).sum() * TIMESTEP_HOURS
        drift_rows = [
            ('--- Forecast Drift (simulated − forecast) ---', '', ''),
            ('Cumulative drift', f'{drift_kwh:+,.0f}',  'kWh'),
            ('Mean absolute error', f'{mae_kw:.1f}',    'kW per 5-min step'),
            ('Hours below forecast', f'{h_below:.1f}',  'h  (recovery active)'),
            ('Hours above forecast', f'{h_above:.1f}',  'h'),
            ('', '', ''),
        ]

    summary_rows = [
        ('--- Profit / Cost ---', '', ''),
        ('Reference profit',   f'{total_ref:+.2f}',   'CHF'),
        ('Optimised profit',   f'{total_opt:+.2f}',   'CHF'),
        ('Optimizer gain',     f'{total_gain:+.2f}',  'CHF'),
        ('Days opt better',    days_opt_better,        ''),
        ('Days ref better',    days_ref_better,        ''),
        ('', '', ''),
        ('--- Energy Production ---', '', ''),
        ('Reference energy',   f'{ref_energy_kwh:,.0f}',  'kWh'),
        ('Optimised energy',   f'{opt_energy_kwh:,.0f}',  'kWh'),
        ('Difference',         f'{energy_diff:+,.0f}',    'kWh'),
        ('', '', ''),
        *drift_rows,
        ('--- Spillage Losses ---', '', ''),
        ('Bidmi spillage',     f'{total_spill_b:,.0f}',  'kWh'),
        ('Haselholz spillage', f'{total_spill_h:,.0f}',  'kWh'),
        ('Total spillage',     f'{total_spill:,.0f}',    'kWh'),
        ('', '', ''),
        ('--- Model ---', '', ''),
        ('Mode', MODE, ''),
    ]

    summary_df = pd.DataFrame(summary_rows, columns=['Metric', 'Value', 'Unit'])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        opt_df.to_excel(writer, sheet_name='Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nSaved: {output_path}")

    # Console summary
    print(f"\n{'─'*55}")
    print(f"  Mode              : {MODE}")
    print(f"  Reference profit  : {total_ref:+,.2f} CHF")
    print(f"  Optimised profit  : {total_opt:+,.2f} CHF")
    print(f"  Gain              : {total_gain:+,.2f} CHF")
    print(f"  Ref energy        : {ref_energy_kwh/1000:,.1f} MWh")
    print(f"  Opt energy        : {opt_energy_kwh/1000:,.1f} MWh")
    print(f"  Total spill loss  : {total_spill/1000:,.1f} MWh")
    if drift_rows:
        print(f"  Forecast MAE      : {mae_kw:.1f} kW")
        print(f"  Cumulative drift  : {drift_kwh/1000:+,.1f} MWh")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root      = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, DATA_FILENAME)

    if not os.path.exists(data_path):
        print(f"ERROR: '{DATA_FILENAME}' not found in {root}")
        sys.exit(1)

    target_date = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Mode : {MODE}")
    print(f"Loading: {data_path}")
    df = load_data(data_path)
    print(f"  {len(df)} rows, {df['DateTime'].dt.date.nunique()} days")
    if target_date:
        print(f"  Single day: {target_date}")
    print()

    opt_df = run_all(df, target_date)

    output_path = os.path.join(root, 'Output', OUTPUT_FILENAME)
    save_output(opt_df, output_path)


if __name__ == '__main__':
    main()
