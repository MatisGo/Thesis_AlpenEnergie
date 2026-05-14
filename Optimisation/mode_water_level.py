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

  ID component  = -|Grid_export - Forecast| × ID_price × dt / 1000  [EUR]
                  Absolute deviation from forecast always costs money.
                  When battery INTRADAY mode is active, grid export = turbine +
                  battery_net, so the battery can reduce the ID cost.
"""

from hydro_constants import (
    P_MAX_M2, P_MAX_M1, TIMESTEP_HOURS,
    BIDMI_LEVEL_MIN, HASELHOLZ_LEVEL_MIN,
    BIDMI_RANGE, HASELHOLZ_RANGE,
    water_balance_step, attach_common_results,
)
from battery_control import battery_step_forecast


def dispatch_day(day_df, lb0, lh0, target_lb, target_lh, battery_cfg=None,
                 seasonal_prod=None, **_kwargs):
    """
    battery_cfg : BatteryConfig or None.
      If mode == 'INTRADAY', battery is co-simulated at each timestep
      (same architecture as mode_forecast.py).
    """
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    demand        = day_df['Consumption_kW'].tolist()
    forecast_cons = day_df['Forecast_Consumption_kW'].tolist()
    da_price      = day_df['Day_Ahead_Price_EUR_MWh'].tolist()
    bg_long_r     = day_df['BG_Long_EUR_MWh'].tolist()   # real prices for settlement
    bg_short_r    = day_df['BG_Short_EUR_MWh'].tolist()
    forecast      = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    co_sim = battery_cfg is not None and battery_cfg.mode in ('INTRADAY', 'HYBRID')
    soc    = battery_cfg.soc0 if co_sim else 0.0

    opt_m2, opt_m1           = [], []
    opt_lb, opt_lh           = [], []
    opt_spill_b, opt_spill_h = [], []
    batt_c_list, batt_d_list, batt_soc_list = [], [], []

    lb, lh = lb0, lh0

    for t in range(N):
        norm_lb = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
        norm_lh = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE

        # Battery co-simulation: decide before turbine runs, adjust target
        # For HYBRID, soc_max_override caps INTRADAY zone below the DA block floor.
        if co_sim:
            batt_soc_list.append(soc)
            batt_c, batt_d, soc = battery_step_forecast(soc, norm_lb, norm_lh)
        else:
            batt_c = batt_d = 0.0
        batt_c_list.append(batt_c)
        batt_d_list.append(batt_d)

        # Follow seasonal average production (30-day rolling avg), falling back
        # to raw reference if seasonal data is unavailable for this day.
        # Charging → turbine produces target + charge (releases more water)
        # Discharging → turbine produces target − discharge (saves water)
        if seasonal_prod is not None and t < len(seasonal_prod[0]):
            ref_m2 = max(0.0, min(P_MAX_M2, float(seasonal_prod[0][t])))
            ref_m1 = max(0.0, min(P_MAX_M1, float(seasonal_prod[1][t])))
        else:
            ref_m2 = max(0.0, min(P_MAX_M2, float(day_df.loc[t, 'Ref_M2_kW'])))
            ref_m1 = max(0.0, min(P_MAX_M1, float(day_df.loc[t, 'Ref_M1_kW'])))
        ref_total = ref_m2 + ref_m1

        if co_sim and (batt_c > 0 or batt_d > 0):
            # Scale both turbines proportionally to reach adjusted target
            target_total = max(0.0, ref_total + batt_c - batt_d)
            scale = target_total / ref_total if ref_total > 0 else 0.0
            m2 = max(0.0, min(P_MAX_M2, ref_m2 * scale))
            m1 = max(0.0, min(P_MAX_M1, ref_m1 * scale))
        else:
            m2, m1 = ref_m2, ref_m1

        lb, lh, spill_b, spill_h, m2, m1 = water_balance_step(lb, lh, m2, m1, inflow_b[t], inflow_h[t])

        opt_m2.append(m2);  opt_m1.append(m1)
        opt_lb.append(lb);  opt_lh.append(lh)
        opt_spill_b.append(spill_b); opt_spill_h.append(spill_h)

    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, target_lb, target_lh,
        mode_name='WATER_LEVEL',
    )

    # DA bid = Forecast_Production - Forecast_Consumption (net position, no look-ahead)
    da_bid_list  = [forecast[t] - forecast_cons[t] for t in range(N)]
    da_component = [
        da_bid_list[t] * TIMESTEP_HOURS * da_price[t] / 1000.0
        for t in range(N)]

    if co_sim:
        grid_out  = [opt_m2[t] + opt_m1[t] + batt_d_list[t] - batt_c_list[t]
                     for t in range(N)]
        dev_total = [(grid_out[t] - demand[t]) - da_bid_list[t] for t in range(N)]
        # Split the metered (total) imbalance into long/short components
        id_long = [
            max(0.0, dev_total[t]) * bg_long_r[t]  * TIMESTEP_HOURS / 1000.0
            for t in range(N)]
        id_short = [
            min(0.0, dev_total[t]) * bg_short_r[t] * TIMESTEP_HOURS / 1000.0
            for t in range(N)]
        id_component = [id_long[t] + id_short[t] for t in range(N)]
        dev_hydro = [(opt_m2[t] + opt_m1[t] - demand[t]) - da_bid_list[t] for t in range(N)]
        id_hydro_only = [
            dev_hydro[t] * (bg_long_r[t] if dev_hydro[t] >= 0 else bg_short_r[t]) * TIMESTEP_HOURS / 1000.0
            for t in range(N)]
        day_df['Batt_Charge_kW']    = batt_c_list
        day_df['Batt_Discharge_kW'] = batt_d_list
        day_df['Batt_SOC_kWh']      = batt_soc_list
        day_df['Batt_Net_kW']       = [batt_d_list[t] - batt_c_list[t] for t in range(N)]
        day_df['Batt_Revenue_EUR']  = [id_component[t] - id_hydro_only[t] for t in range(N)]
        day_df._batt_soc_end        = soc
    else:
        dev = [(opt_m2[t] + opt_m1[t] - demand[t]) - da_bid_list[t] for t in range(N)]
        id_long = [
            max(0.0, dev[t]) * bg_long_r[t]  * TIMESTEP_HOURS / 1000.0
            for t in range(N)]
        id_short = [
            min(0.0, dev[t]) * bg_short_r[t] * TIMESTEP_HOURS / 1000.0
            for t in range(N)]
        id_component = [id_long[t] + id_short[t] for t in range(N)]

    day_df['Opt_DA_Trading_EUR']      = da_component
    day_df['Opt_Imbalance_Long_EUR']  = id_long
    day_df['Opt_Imbalance_Short_EUR'] = id_short
    day_df['Opt_ID_Imbalance_EUR']    = id_component
    day_df['Opt_Energy_Trading_EUR']  = [da_component[t] + id_component[t] for t in range(N)]
    day_df['Forecast_Drift_kW']      = [
        (opt_m2[t] + opt_m1[t]) - forecast[t] for t in range(N)]

    return day_df
