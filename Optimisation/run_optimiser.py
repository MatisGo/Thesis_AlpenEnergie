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
    load_data, compute_seasonal_targets, compute_seasonal_production,
)

# ===========================================================================
#  SELECT MODE HERE
# ===========================================================================
#
#   'FORECAST'      Rule-based 3-state equalization (follows CNN/LSTM forecast)
#   'WATER_VALUE'   LP that minimises intraday imbalance costs
#   'WATER_LEVEL'   Follows historical reference (seasonal average) production
#
MODE = 'WATER_VALUE'

# ---------------------------------------------------------------------------
#  BATTERY SETTINGS
# ---------------------------------------------------------------------------
#
#   BATTERY_ACTIVE  : True  — run battery dispatch after hydro each day
#                     False — hydro only
#   BATTERY_MODE    : 'DAY_AHEAD' — buy low / sell high on the DA market (LP)
#                     'INTRADAY'  — reservoir-triggered co-simulation (reduces intraday imbalance)
#                     'HYBRID'    — INTRADAY + optional DA block when SOC > floor & price OK
#
BATTERY_ACTIVE = True
BATTERY_MODE   = 'INTRADAY'  # 'DAY_AHEAD', 'INTRADAY', 'HYBRID'

# ---------------------------------------------------------------------------
#  FINANCIAL PARAMETERS  (battery investment analysis)
# ---------------------------------------------------------------------------
#
#   BATTERY_COST_PER_KWH   : 500 EUR/kWh  — Needs to be updated with real market data for accurate analysis.
#   BATTERY_LIFETIME_YEARS : 15 years     — assumed useful life
#   DISCOUNT_RATE          : 0.05         — 5% annual discount rate // NEEDS to be checked
#   BATTERY_DEG_PER_CYCLE  : fraction of capacity lost per full cycle
#                            e.g. 0.0001 = 0.01 %/cycle → 80 % remaining after 2 000 cycles
#   BATTERY_OM_EUR_KWH_YEAR: EUR/kWh/year — O&M cost per unit of installed capacity
#                            e.g. 8.0 EUR/kWh/year → 8000 EUR/year for a 1000 kWh battery
#
BATTERY_COST_PER_KWH    = 300.0    # EUR/kWh
BATTERY_LIFETIME_YEARS  =  15      # years
DISCOUNT_RATE           = 0.05     # —
BATTERY_DEG_PER_CYCLE   = 0.00005    # fraction/cycle  (0.005 %/cycle)
BATTERY_OM_EUR_KWH_YEAR = 8.0     # EUR/kWh/year (8 could be default)

# ---------------------------------------------------------------------------
#  HYBRID BATTERY MODE TUNING  (active only when BATTERY_MODE = 'HYBRID')
# ---------------------------------------------------------------------------
#
#   HYBRID_BLOCK_FILL_PCT : fraction of the usable SOC range [SOC_MIN .. SOC_MAX]
#                           above which the DA block is triggered.
#                           e.g. 0.80 with [20, 180] kWh → floor = 20 + 0.80×160 = 148 kWh
#                           Energy above the floor is reserved for DA delivery.
#
#   HYBRID_MIN_DA_PRICE   : minimum next-day PEAK DA price [EUR/MWh] required
#                           to activate the block.  Below this threshold the
#                           battery stays in pure INTRADAY mode for that day.
#
#   HYBRID_BID_HOUR       : Hour of day (0–23) at which the SOC check is
#                           performed and the DA bid is locked in.
#                           Default = 12 (noon) — matches the real DA auction.
#
HYBRID_BLOCK_FILL_PCT = 0.80     # fraction of usable range
HYBRID_MIN_DA_PRICE   = 50.0     # EUR/MWh
HYBRID_BID_HOUR       = 8       # 0-23
# ===========================================================================

def _make_output_filename():
    """Build a short, descriptive filename from run parameters."""
    mode_short = {'FORECAST': 'FC', 'WATER_VALUE': 'WV', 'WATER_LEVEL': 'WL'}.get(MODE, MODE)
    if not BATTERY_ACTIVE:
        batt_str = 'NoBatt'
    else:
        batt_short = {'DAY_AHEAD': 'DA', 'INTRADAY': 'ID', 'HYBRID': 'HYB'}.get(BATTERY_MODE, BATTERY_MODE)
        mwh = int(round(CAPACITY_KWH / 1000))
        batt_str = f'{batt_short}_{mwh}MWh'
    return f'{mode_short}_{batt_str}.xlsx'

# Import the correct mode module
if MODE == 'FORECAST':
    from mode_forecast import dispatch_day
elif MODE == 'WATER_VALUE':
    from mode_water_value import dispatch_day, get_solver
elif MODE == 'WATER_LEVEL':
    from mode_water_level import dispatch_day
else:
    raise ValueError(f"Unknown MODE: '{MODE}'")

if BATTERY_ACTIVE:
    from battery_control import (
        BatteryConfig,
        dispatch_day_battery, get_battery_solver,
        SOC_INITIAL, CYCLE_KWH, CAPACITY_KWH, ROUND_TRIP_EFF, C_RATE,
        SOC_MIN, SOC_MAX, EFF_DISCHARGE, _P_MAX_BATT,
    )


# ---------------------------------------------------------------------------
# HYBRID helpers
# ---------------------------------------------------------------------------

def _get_da_floor():
    """DA block floor SOC [kWh]: INTRADAY zone ceiling when a block is active."""
    return SOC_MIN + HYBRID_BLOCK_FILL_PCT * (SOC_MAX - SOC_MIN)


def _apply_da_delivery(day_df, commit_kwh, peak_t):
    """
    Apply committed DA energy delivery starting at peak_t.

    commit_kwh : kWh stored in battery (SOC units) reserved for DA sale.
                 Grid receives commit_kwh × EFF_DISCHARGE kWh.
    peak_t     : timestep index with the highest DA price (delivery start).

    Adds to Batt_Discharge_kW, Batt_Net_kW, and Batt_Revenue_EUR.
    The SOC column is NOT modified — it reflects the INTRADAY zone only.

    Returns grid kWh delivered (added to cycle counter).
    """
    dt             = TIMESTEP_HOURS
    n              = len(day_df)
    grid_kwh_total = commit_kwh * EFF_DISCHARGE       # kWh sent to grid
    remaining_kwh  = grid_kwh_total

    t = peak_t
    while remaining_kwh > 1e-6 and t < n:
        grid_kw    = min(_P_MAX_BATT, remaining_kwh / dt)
        da_price_t = float(day_df.loc[t, 'Day_Ahead_Price_EUR_MWh'])
        revenue    = grid_kw * da_price_t * dt / 1000.0

        day_df.loc[t, 'Batt_Discharge_kW'] = float(day_df.loc[t, 'Batt_Discharge_kW']) + grid_kw
        day_df.loc[t, 'Batt_Net_kW']       = float(day_df.loc[t, 'Batt_Net_kW'])       + grid_kw
        day_df.loc[t, 'Batt_Revenue_EUR']  = float(day_df.loc[t, 'Batt_Revenue_EUR'])  + revenue

        remaining_kwh -= grid_kw * dt
        t += 1

    return grid_kwh_total


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
    """Iterate over all days, chain reservoir levels, return full result DataFrame."""
    import datetime as _dt

    print("Computing seasonal targets...")
    season_targets    = compute_seasonal_targets(df)
    season_production = compute_seasonal_production(df) if MODE == 'WATER_LEVEL' else {}

    solver         = get_solver() if MODE == 'WATER_VALUE' else None
    batt_solver    = get_battery_solver() if BATTERY_ACTIVE else None
    prev_soc       = SOC_INITIAL if BATTERY_ACTIVE else None
    total_discharged_kwh = 0.0

    days = sorted(df['DateTime'].dt.date.unique())
    if target_date:
        days = [d for d in days if str(d) == target_date]
        if not days:
            print(f"ERROR: date '{target_date}' not found in data.")
            sys.exit(1)

    days_set    = set(days)
    results     = []
    prev_lb     = None
    prev_lh     = None
    failed_days = 0
    da_block             = {}    # date → {'commit_kwh': float, 'peak_t': int}  (HYBRID only)
    total_da_delivery_kwh = 0.0  # kWh actually delivered via DA blocks
    da_block_days         = 0    # number of days a DA block was executed

    for i, day in enumerate(days, 1):
        day_df = df[df['DateTime'].dt.date == day].copy()

        t_lb, t_lh = season_targets.get(day, (None, None))

        if t_lb is None: t_lb = float(day_df['Bidmi_mm'].iloc[0])
        if t_lh is None: t_lh = float(day_df['Haselholz_mm'].iloc[0])

        lb0 = prev_lb if prev_lb is not None else float(day_df['Bidmi_mm'].iloc[0])
        lh0 = prev_lh if prev_lh is not None else float(day_df['Haselholz_mm'].iloc[0])

        # Build battery config for this day (carries today's starting SOC).
        # INTRADAY / HYBRID battery is co-simulated inside dispatch_day.
        # DAY_AHEAD battery is fully independent → dispatch_day ignores it.
        # HYBRID with active block: INTRADAY zone is capped at DA floor so
        #   committed energy stays reserved until delivery.
        if BATTERY_ACTIVE:
            if BATTERY_MODE == 'HYBRID':
                block_today = da_block.get(day)
                if block_today:
                    da_floor      = _get_da_floor()
                    soc0_intraday = min(prev_soc, da_floor)
                    battery_cfg   = BatteryConfig(mode='HYBRID', soc0=soc0_intraday,
                                                  soc_max_override=da_floor)
                else:
                    battery_cfg = BatteryConfig(mode='HYBRID', soc0=prev_soc,
                                                soc_max_override=None)
            else:
                battery_cfg = BatteryConfig(mode=BATTERY_MODE, soc0=prev_soc)
        else:
            battery_cfg = None

        seas_prod = season_production.get(day) if MODE == 'WATER_LEVEL' else None

        day_df = dispatch_day(
            day_df, lb0, lh0, t_lb, t_lh,
            solver=solver,
            battery_cfg=battery_cfg,
            seasonal_prod=seas_prod,
        )

        # Read hydro chain attributes before any reset_index can drop them
        prev_lb = getattr(day_df, '_opt_lb_end', None)
        prev_lh = getattr(day_df, '_opt_lh_end', None)
        dev_b   = getattr(day_df, '_dev_b',      float('nan'))
        dev_h   = getattr(day_df, '_dev_h',      float('nan'))

        # Battery post-dispatch
        batt_str = ""
        if BATTERY_ACTIVE:
            if BATTERY_MODE in ('INTRADAY', 'HYBRID'):
                # Co-simulation already done inside dispatch_day.
                prev_soc       = getattr(day_df, '_batt_soc_end', prev_soc)
                discharged_kwh = day_df['Batt_Discharge_kW'].sum() * TIMESTEP_HOURS

                if BATTERY_MODE == 'HYBRID':
                    # 1. Apply any DA delivery committed from yesterday
                    block_today = da_block.get(day)
                    if block_today:
                        extra_kwh              = _apply_da_delivery(
                            day_df, block_today['commit_kwh'], block_today['peak_t'])
                        discharged_kwh        += extra_kwh
                        total_da_delivery_kwh += extra_kwh
                        da_block_days         += 1
                        # prev_soc = INTRADAY end (committed zone delivered, already excluded
                        # from INTRADAY dispatch via soc_max_override)

                    # 2. Check if conditions for a new DA block (delivery tomorrow) are met
                    tomorrow = day + _dt.timedelta(days=1)
                    if tomorrow in days_set and tomorrow not in da_block:
                        bid_t = HYBRID_BID_HOUR * (60 // 5)   # 12 × 12 = 144 for noon
                        if bid_t < len(day_df):
                            soc_at_bid = float(day_df['Batt_SOC_kWh'].iloc[bid_t])
                            da_floor   = _get_da_floor()
                            if soc_at_bid > da_floor:
                                next_day_df = df[df['DateTime'].dt.date == tomorrow]
                                next_prices = next_day_df['Day_Ahead_Price_EUR_MWh'].tolist()
                                if next_prices:
                                    peak_price = max(next_prices)
                                    if peak_price >= HYBRID_MIN_DA_PRICE:
                                        commit_kwh = soc_at_bid - da_floor
                                        peak_t     = int(pd.Series(next_prices).idxmax())
                                        da_block[tomorrow] = {
                                            'commit_kwh': commit_kwh,
                                            'peak_t':     peak_t,
                                        }
                                        print(f"  [HYBRID] Block: {commit_kwh:.1f} kWh → "
                                              f"{tomorrow}  peak {peak_price:.0f} EUR/MWh t={peak_t}")

            else:  # DAY_AHEAD — fully independent LP, run now
                day_df, prev_soc, discharged_kwh = dispatch_day_battery(
                    day_df, prev_soc, batt_solver, mode=BATTERY_MODE)

            total_discharged_kwh += discharged_kwh
            cycles_today = discharged_kwh / CYCLE_KWH
            batt_rev     = day_df['Batt_Revenue_EUR'].sum()
            batt_str     = f"  batt {batt_rev:+.2f} EUR  cyc {cycles_today:.2f}"

        # Console progress
        opt_profit = day_df['Opt_Energy_Trading_EUR'].sum()
        ref_profit = day_df['Ref_Energy_Trading_EUR'].sum()
        gain       = opt_profit - ref_profit
        spill      = day_df['Opt_Spill_Bidmi_kWh'].sum() + day_df['Opt_Spill_Haselholz_kWh'].sum()
        spill_str  = f"  SPILL {spill:.0f} kWh" if spill > 0.1 else ""

        print(f"[{i:3}/{len(days)}] {day}  "
              f"tgt B={t_lb:.0f} H={t_lh:.0f}  "
              f"ref {ref_profit:+.2f} EUR  opt {opt_profit:+.2f} EUR  "
              f"gain {gain:+.2f} EUR  "
              f"dLB={dev_b:+.1f}mm dLH={dev_h:+.1f}mm"
              f"{spill_str}{batt_str}")

        failed_days += getattr(day_df, '_failed', 0)
        results.append(day_df)

    result_df = pd.concat(results, ignore_index=True)
    result_df._total_discharged_kwh  = total_discharged_kwh
    result_df._failed_days           = failed_days
    result_df._da_delivery_kwh       = total_da_delivery_kwh
    result_df._da_block_days         = da_block_days
    return result_df


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_output(opt_df: pd.DataFrame, output_path: str):
    """Save Results + Summary sheets to Excel."""
    opt_df['Opt_Total_Energy_Cost_EUR'] = opt_df['Opt_Energy_Trading_EUR'].cumsum()

    total_ref  = opt_df['Ref_Energy_Trading_EUR'].sum()
    total_opt  = opt_df['Opt_Energy_Trading_EUR'].sum()
    total_gain = total_opt - total_ref

    daily_ref        = opt_df.groupby(opt_df['DateTime'].dt.date)['Ref_Energy_Trading_EUR'].sum()
    daily_opt        = opt_df.groupby(opt_df['DateTime'].dt.date)['Opt_Energy_Trading_EUR'].sum()
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

    # Battery summary
    batt_rows = []
    if BATTERY_ACTIVE and 'Batt_Revenue_EUR' in opt_df.columns:
        total_batt_rev    = opt_df['Batt_Revenue_EUR'].sum()
        total_discharged  = getattr(opt_df, '_total_discharged_kwh', 0.0)
        total_cycles      = total_discharged / CYCLE_KWH
        total_charged_kwh = (opt_df['Batt_Charge_kW'] * TIMESTEP_HOURS).sum()
        da_delivery_kwh   = getattr(opt_df, '_da_delivery_kwh', 0.0)
        da_block_days     = getattr(opt_df, '_da_block_days',   0)
        da_pct            = (da_delivery_kwh / total_discharged * 100
                             if total_discharged > 0 else 0.0)
        # Financial analysis
        n_days            = opt_df['DateTime'].dt.date.nunique()
        batt_capex        = CAPACITY_KWH * BATTERY_COST_PER_KWH
        annual_revenue    = total_batt_rev * 365.0 / n_days
        annual_om         = BATTERY_OM_EUR_KWH_YEAR * CAPACITY_KWH
        annual_cf         = annual_revenue - annual_om
        annual_cycles_sim = total_cycles * 365.0 / n_days   # annualised from simulation

        # Capacity factor in year t: linear degradation per cycle.
        # cap_factor(t) = max(0, 1 - DEG_PER_CYCLE × cumulative_cycles_at_end_of_year_t)
        # cumulative cycles at end of year t = annual_cycles × t
        def cap_factor(t):
            return max(0.0, 1.0 - BATTERY_DEG_PER_CYCLE * annual_cycles_sim * t)

        # NPV: cash flow in year t = annual_cf × cap_factor(t), discounted at DISCOUNT_RATE
        npv               = -batt_capex + sum(
                                annual_cf * cap_factor(t) / (1 + DISCOUNT_RATE) ** t
                                for t in range(1, BATTERY_LIFETIME_YEARS + 1))
        simple_payback    = (batt_capex / annual_cf) if annual_cf > 0 else float('inf')

        # Lifetime kWh discharged also degrades each year with cap_factor
        annual_discharged = total_discharged * 365.0 / n_days
        lifetime_kwh      = sum(
                                annual_discharged * cap_factor(t)
                                for t in range(1, BATTERY_LIFETIME_YEARS + 1))
        lcos              = (batt_capex / lifetime_kwh) if lifetime_kwh > 0 else float('inf')
        capacity_end_life = round(cap_factor(BATTERY_LIFETIME_YEARS) * 100, 1)

        batt_rows = [
            ('--- Battery (BESS) ---', '', ''),
            ('Mode',                  BATTERY_MODE,                    ''),
            ('Capacity',              f'{CAPACITY_KWH:.1f}',           'kWh'),
            ('Round-trip efficiency', f'{ROUND_TRIP_EFF*100:.0f}',     '%'),
            ('Total charged',         f'{total_charged_kwh:,.1f}',     'kWh'),
            ('Total discharged',      f'{total_discharged:,.1f}',      'kWh'),
            ('Total cycles',          f'{total_cycles:.1f}',           'cycles'),
            ('Battery revenue',       f'{total_batt_rev:+,.2f}',       'EUR'),
            ('DA block days',         f'{da_block_days}  ({da_pct:.1f}% of discharge)',  'days'),
            ('', '', ''),
            ('--- Battery Investment ---', '', ''),
            ('Days simulated',        f'{n_days}',                                     'days'),
            ('Annual cycles',         f'{annual_cycles_sim:.1f}',                      'cycles/year'),
            ('Annualised revenue',    f'{annual_revenue:+,.2f}',                        'EUR/year'),
            ('O&M costs',             f'{BATTERY_OM_EUR_KWH_YEAR:.1f} EUR/kWh/yr  →  {annual_om:,.0f}',  'EUR/year'),
            ('Annual cash flow',      f'{annual_cf:+,.2f}',                            'EUR/year'),
            (f'CAPEX  ({BATTERY_COST_PER_KWH:.0f} EUR/kWh)',
                                      f'{batt_capex:,.0f}',                            'EUR'),
            (f'NPV  ({BATTERY_LIFETIME_YEARS}y, {DISCOUNT_RATE*100:.0f}%, {BATTERY_DEG_PER_CYCLE*100:.4f}%/cyc)',
                                      f'{npv:+,.0f}',                                  'EUR'),
            ('Simple payback',        f'{simple_payback:.1f}' if simple_payback < 100 else '>100',
                                                                                        'years'),
            ('LCOS',                  f'{lcos:.3f}' if lcos < 9999 else 'N/A',         'EUR/kWh cycled'),
            (f'Capacity at yr {BATTERY_LIFETIME_YEARS}',
                                      f'{capacity_end_life:.1f}',                      '%'),
            ('', '', ''),
        ]

    summary_rows = [
        ('--- Profit / Cost ---', '', ''),
        ('Reference profit',   f'{total_ref:+.2f}',   'EUR'),
        ('Optimised profit',   f'{total_opt:+.2f}',   'EUR'),
        ('Optimizer gain',     f'{total_gain:+.2f}',  'EUR'),
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
        *batt_rows,
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
    print(f"  Reference profit  : {total_ref:+,.2f} EUR")
    print(f"  Optimised profit  : {total_opt:+,.2f} EUR")
    print(f"  Gain              : {total_gain:+,.2f} EUR")
    print(f"  Ref energy        : {ref_energy_kwh/1000:,.1f} MWh")
    print(f"  Opt energy        : {opt_energy_kwh/1000:,.1f} MWh")
    print(f"  Total spill loss  : {total_spill/1000:,.1f} MWh")
    if drift_rows:
        print(f"  Forecast MAE      : {mae_kw:.1f} kW")
        print(f"  Cumulative drift  : {drift_kwh/1000:+,.1f} MWh")
    if batt_rows:
        print(f"  {'─'*51}")
        print(f"  Battery revenue   : {total_batt_rev:+,.2f} EUR  ({n_days}d → {annual_revenue:+,.0f} EUR/yr)")
        print(f"  Battery cycles    : {total_cycles:.1f}  ({total_discharged/1000:,.1f} MWh discharged)")
        total_with_batt = total_opt + total_batt_rev
        print(f"  Total (hydro+bat) : {total_with_batt:+,.2f} EUR")
        print(f"  CAPEX             : {batt_capex:,.0f} EUR")
        print(f"  NPV ({BATTERY_LIFETIME_YEARS}y,{DISCOUNT_RATE*100:.0f}%)     : {npv:+,.0f} EUR")
        print(f"  Simple payback    : {simple_payback:.1f} years" if simple_payback < 100
              else f"  Simple payback    : >100 years")
        print(f"  LCOS              : {lcos:.3f} EUR/kWh" if lcos < 9999 else f"  LCOS              : N/A")
    print(f"{'─'*55}")

    # Return a flat summary dict for Main_results.xlsx
    return {
        'Mode':               MODE,
        'Battery_Active':     BATTERY_ACTIVE,
        'Battery_Mode':       BATTERY_MODE            if BATTERY_ACTIVE else '-',
        'Capacity_kWh':       CAPACITY_KWH            if BATTERY_ACTIVE else 0,
        'C_Rate':             C_RATE                  if BATTERY_ACTIVE else '-',
        'RTE_pct':            round(ROUND_TRIP_EFF * 100) if BATTERY_ACTIVE else '-',
        'Deg_pct_per_cycle':  BATTERY_DEG_PER_CYCLE * 100  if BATTERY_ACTIVE else '-',
        'Lifetime_Years':     BATTERY_LIFETIME_YEARS  if BATTERY_ACTIVE else '-',
        'Discount_Rate_pct':  DISCOUNT_RATE * 100     if BATTERY_ACTIVE else '-',
        'OM_EUR_kWh_yr':      BATTERY_OM_EUR_KWH_YEAR  if BATTERY_ACTIVE else '-',
        'OM_Cost_EUR_Year':   annual_om                if batt_rows else 0,
        'Ref_Profit_EUR':  round(total_ref, 2),
        'Opt_Profit_EUR':  round(total_opt, 2),
        'Gain_EUR':        round(total_gain, 2),
        'Days_Opt_Better': int(days_opt_better),
        'Days_Ref_Better': int(days_ref_better),
        'Opt_Energy_MWh':  round(opt_energy_kwh / 1000, 1),
        'Spill_MWh':       round(total_spill / 1000, 1),
        'Forecast_MAE_kW': round(mae_kw, 1) if drift_rows else '-',
        'Batt_Revenue_EUR':    round(total_batt_rev, 2)    if batt_rows else 0,
        'Batt_Cycles':         round(total_cycles, 1)      if batt_rows else 0,
        'Annual_Cycles':       round(annual_cycles_sim, 1) if batt_rows else 0,
        'Days_Simulated':      n_days                      if batt_rows else opt_df['DateTime'].dt.date.nunique(),
        'Annual_Revenue_EUR':  round(annual_revenue, 2)    if batt_rows else 0,
        'CAPEX_EUR':           round(batt_capex, 0)        if batt_rows else 0,
        'NPV_EUR':             round(npv, 0)               if batt_rows else 0,
        'Payback_Years':       round(simple_payback, 1)    if (batt_rows and simple_payback < 100) else ('>100' if batt_rows else 0),
        'LCOS_EUR_kWh':        round(lcos, 3)              if (batt_rows and lcos < 9999) else 0,
        'Capacity_EOL_pct':    capacity_end_life           if batt_rows else 0,
        'DA_Block_Days':        da_block_days           if batt_rows else '-',
        'DA_Delivery_Pct':     round(da_pct, 1)        if batt_rows else '-',
        'Failed_Days':         getattr(opt_df, '_failed_days', 0),
        'Output_File':         os.path.basename(output_path),
    }


# ---------------------------------------------------------------------------
# Main results log  (one row per run, appended to Main_results.xlsx)
# ---------------------------------------------------------------------------

def append_main_results(output_dir: str, summary: dict):
    """Append one summary row to Output/Main_results.xlsx."""
    import datetime
    summary = {'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), **summary}
    row_df   = pd.DataFrame([summary])
    path     = os.path.join(output_dir, 'Main_results.xlsx')
    if os.path.exists(path):
        existing = pd.read_excel(path, engine='openpyxl')
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        combined.to_excel(writer, index=False)
    print(f"Main results updated: {path}")


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

    output_dir  = os.path.join(root, 'Output')
    output_path = os.path.join(output_dir, _make_output_filename())
    summary     = save_output(opt_df, output_path)

    # Skip Main_results when running a single day (partial data)
    if not target_date:
        append_main_results(output_dir, summary)


if __name__ == '__main__':
    main()
