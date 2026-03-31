"""
mode_water_level.py
===================
WATER_LEVEL mode — production follows the seasonal (historical reference) average.

At each timestep the turbines reproduce the historical reference production
(Ref_M2_kW + Ref_M1_kW from the data file).  This represents the long-run
seasonal average operation and serves as a simple baseline for comparison.

Pricing (same structure as FORECAST and WATER_VALUE modes)
----------------------------------------------------------
  DA component  = (Forecast - Consumption) × DA_price × dt / 1000   [EUR]
                  Fixed scheduled position submitted to the day-ahead market.

  ID component  = -|Production - Forecast| × ID_price × dt / 1000  [EUR]
                  Absolute deviation from forecast always costs money.
"""

from hydro_constants import (
    P_MAX_M2, P_MAX_M1, TIMESTEP_HOURS,
    water_balance_step, attach_common_results,
)


def dispatch_day(day_df, lb0, lh0, target_lb, target_lh, **_kwargs):
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    demand   = day_df['Consumption_kW'].tolist()
    da_price = day_df['Day_Ahead_Price_EUR_MWh'].tolist()
    id_price = day_df['Intra_Day_Price_EUR_MWh'].tolist()
    forecast = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    opt_m2, opt_m1           = [], []
    opt_lb, opt_lh           = [], []
    opt_spill_b, opt_spill_h = [], []

    lb, lh = lb0, lh0

    for t in range(N):
        # Follow historical reference production, clipped to physical limits
        m2 = max(0.0, min(P_MAX_M2, float(day_df.loc[t, 'Ref_M2_kW'])))
        m1 = max(0.0, min(P_MAX_M1, float(day_df.loc[t, 'Ref_M1_kW'])))

        lb, lh, spill_b, spill_h = water_balance_step(lb, lh, m2, m1, inflow_b[t], inflow_h[t])

        opt_m2.append(m2);  opt_m1.append(m1)
        opt_lb.append(lb);  opt_lh.append(lh)
        opt_spill_b.append(spill_b); opt_spill_h.append(spill_h)

    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, da_price, target_lb, target_lh,
        mode_name='WATER_LEVEL',
    )

    # Correct pricing: DA scheduled position + ID imbalance cost
    da_component = [
        (forecast[t] - demand[t]) * TIMESTEP_HOURS * da_price[t] / 1000.0
        for t in range(N)]
    id_component = [
        -abs(opt_m2[t] + opt_m1[t] - forecast[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
        for t in range(N)]

    day_df['Opt_DA_Trading_EUR']   = da_component
    day_df['Opt_ID_Imbalance_EUR'] = id_component
    day_df['Opt_Energy_Trading_EUR'] = [da_component[t] + id_component[t] for t in range(N)]

    day_df['Forecast_Drift_kW'] = [
        (opt_m2[t] + opt_m1[t]) - forecast[t] for t in range(N)]

    return day_df
