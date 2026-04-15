"""
mode_forecast.py
================
FORECAST mode — rule-based dispatch that follows the CNN/LSTM forecast.

The forecast (M1 + M2 combined) is split between the two turbines at every
5-minute timestep using an analytical equalization formula so that both
reservoirs converge toward the same normalised level.

Recovery state machine (3 states)
----------------------------------
  RECOVERY  Both turbines off.
            Triggered when either reservoir drops below RECOVERY_LOW_PCT.

  FILL_H    M2 runs at full power; M1 = 0.
            Bidmi has recovered to RECOVERY_EXIT_PCT.
            M2 cascade actively fills Haselholz.

  NORMAL    Equalization split follows the forecast.
            Haselholz has also recovered to RECOVERY_EXIT_PCT.

Equalization formula (NORMAL state)
-------------------------------------
  Find M2 such that norm_lb_next == norm_lh_next, given M1 = forecast - M2.
  Analytical solution:
      A      = lb + inflow_b / BIDMI_LS_PER_MM - BIDMI_LEVEL_MIN
      B      = lh + inflow_h / HASELHOLZ_LS_PER_MM
               - COEFF_M1_HASELHOLZ × forecast - HASELHOLZ_LEVEL_MIN
      M2_eq  = (HASELHOLZ_RANGE × A - BIDMI_RANGE × B)
               / (BIDMI_RANGE × C2 + HASELHOLZ_RANGE × C1)
  where C1 = COEFF_M2_BIDMI, C2 = COEFF_M2_CASCADE + COEFF_M1_HASELHOLZ.
"""

import pandas as pd
from hydro_constants import (
    P_MAX_M2, P_MAX_M1, RAMP_MAX, RAMP_WINDOW, TIMESTEP_HOURS,
    BIDMI_LEVEL_MIN, HASELHOLZ_LEVEL_MIN,
    BIDMI_RANGE, HASELHOLZ_RANGE,
    COEFF_M2_BIDMI, COEFF_M2_CASCADE, COEFF_M1_HASELHOLZ,
    BIDMI_LS_PER_MM, HASELHOLZ_LS_PER_MM,
    water_balance_step, attach_common_results,
)
from battery_control import battery_step_forecast

# ---------------------------------------------------------------------------
# Recovery thresholds  [fraction of physical range]
# ---------------------------------------------------------------------------
RECOVERY_LOW_PCT  = 0.20   # enter recovery below this level
RECOVERY_EXIT_PCT = 0.50   # Bidmi must reach this to start filling Haselholz
                            # Haselholz must reach this to resume normal mode

# Equalization coefficients
C1 = COEFF_M2_BIDMI                       # mm of Bidmi drop per kW of M2
C2 = COEFF_M2_CASCADE + COEFF_M1_HASELHOLZ  # mm of Haselholz rise per kW of M2


def dispatch_day(day_df, lb0, lh0, target_lb, target_lh, battery_cfg=None, **_kwargs):
    """
    Rule-based dispatch for one day.

    battery_cfg : BatteryConfig or None.
      If mode == 'INTRADAY', the battery is co-simulated at each timestep:
        - battery_step_forecast() decides charge/discharge from current fill levels
        - turbine target adjusts so net grid export stays = Forecast
        - water balance uses the adjusted turbine output → levels are correct
      Results are written as Batt_* columns; _batt_soc_end is stored as an attribute.
      If mode == 'DAY_AHEAD' or None, battery is ignored here (handled separately).
    """
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    demand   = day_df['Consumption_kW'].tolist()
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    co_sim = battery_cfg is not None and battery_cfg.mode in ('INTRADAY', 'HYBRID')
    soc    = battery_cfg.soc0 if co_sim else 0.0

    opt_m2, opt_m1           = [], []
    opt_lb, opt_lh           = [], []
    opt_spill_b, opt_spill_h = [], []
    opt_state                = []
    batt_c_list, batt_d_list, batt_soc_list = [], [], []

    lb, lh = lb0, lh0

    # Initialise recovery state
    norm_lb0 = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
    norm_lh0 = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE
    state = 'RECOVERY' if (norm_lb0 < RECOVERY_LOW_PCT or norm_lh0 < RECOVERY_LOW_PCT) else 'NORMAL'

    for t in range(N):
        forecast_kw = max(0.0, float(day_df.loc[t, 'Forecast_kW']))

        norm_lb = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
        norm_lh = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE

        # ── Battery co-simulation (INTRADAY / HYBRID mode) ───────────────
        # Decision is based on reservoir fill at START of this timestep.
        # Turbine target shifts so that grid export remains = Forecast.
        # For HYBRID, soc_max_override caps the INTRADAY zone below the DA
        # block floor so committed energy stays reserved.
        if co_sim:
            batt_soc_list.append(soc)
            soc_max_eff = battery_cfg.soc_max_override  # None → uses global SOC_MAX
            batt_c, batt_d, soc = battery_step_forecast(soc, norm_lb, norm_lh,
                                                         soc_max_eff=soc_max_eff)
            turbine_target = forecast_kw + batt_c - batt_d
        else:
            batt_c = batt_d = 0.0
            turbine_target = forecast_kw
        batt_c_list.append(batt_c)
        batt_d_list.append(batt_d)

        # ── State transitions ─────────────────────────────────────────────
        if norm_lb < RECOVERY_LOW_PCT or norm_lh < RECOVERY_LOW_PCT:
            state = 'RECOVERY'
        elif state == 'RECOVERY' and norm_lb >= RECOVERY_EXIT_PCT:
            state = 'FILL_H'
        elif state == 'FILL_H' and norm_lh >= RECOVERY_EXIT_PCT:
            state = 'NORMAL'

        # ── Dispatch per state ────────────────────────────────────────────
        if state == 'RECOVERY':
            m2, m1 = 0.0, 0.0

        elif state == 'FILL_H':
            m2 = P_MAX_M2
            if t >= RAMP_WINDOW:
                m2 = max(opt_m2[t - RAMP_WINDOW] - RAMP_MAX,
                         min(opt_m2[t - RAMP_WINDOW] + RAMP_MAX, m2))
            m1 = 0.0

        else:  # NORMAL — equalization split targeting turbine_target
            A = lb + inflow_b[t] / BIDMI_LS_PER_MM - BIDMI_LEVEL_MIN
            B = (lh + inflow_h[t] / HASELHOLZ_LS_PER_MM
                 - COEFF_M1_HASELHOLZ * turbine_target - HASELHOLZ_LEVEL_MIN)
            m2_eq = (HASELHOLZ_RANGE * A - BIDMI_RANGE * B) / (BIDMI_RANGE * C2 + HASELHOLZ_RANGE * C1)

            m2 = max(0.0, min(P_MAX_M2, m2_eq, turbine_target))
            if t >= RAMP_WINDOW:
                m2 = max(opt_m2[t - RAMP_WINDOW] - RAMP_MAX,
                         min(opt_m2[t - RAMP_WINDOW] + RAMP_MAX, m2))

            m1_target = turbine_target - m2
            m1 = max(0.0, min(P_MAX_M1, m1_target))
            if t >= RAMP_WINDOW:
                m1 = max(opt_m1[t - RAMP_WINDOW] - RAMP_MAX,
                         min(opt_m1[t - RAMP_WINDOW] + RAMP_MAX, m1))

        # Water balance uses ACTUAL turbine output — levels are always correct
        lb, lh, spill_b, spill_h = water_balance_step(lb, lh, m2, m1, inflow_b[t], inflow_h[t])

        opt_m2.append(m2);  opt_m1.append(m1)
        opt_lb.append(lb);  opt_lh.append(lh)
        opt_spill_b.append(spill_b); opt_spill_h.append(spill_h)
        opt_state.append(state)

    da_price      = day_df['Day_Ahead_Price_EUR_MWh'].tolist()
    id_price      = day_df['Intra_Day_Price_EUR_MWh'].tolist()
    forecast_list = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]

    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, target_lb, target_lh,
        mode_name='FORECAST',
    )

    # DA component (fixed — based on forecast bid, same for all modes)
    da_component = [
        (forecast_list[t] - demand[t]) * TIMESTEP_HOURS * da_price[t] / 1000.0
        for t in range(N)]

    if co_sim:
        # Battery was co-simulated: grid export = turbine + battery_net ≈ Forecast
        # ID imbalance is computed on actual GRID output, not turbine alone.
        grid_out  = [opt_m2[t] + opt_m1[t] + batt_d_list[t] - batt_c_list[t]
                     for t in range(N)]
        id_component = [
            -abs(grid_out[t] - forecast_list[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]

        # Battery revenue = ID cost saved vs hydro-only (grid closer to forecast)
        id_hydro_only = [
            -abs(opt_m2[t] + opt_m1[t] - forecast_list[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]
        day_df['Batt_Charge_kW']    = batt_c_list
        day_df['Batt_Discharge_kW'] = batt_d_list
        day_df['Batt_SOC_kWh']      = batt_soc_list
        day_df['Batt_Net_kW']       = [batt_d_list[t] - batt_c_list[t] for t in range(N)]
        day_df['Batt_Revenue_EUR']  = [id_component[t] - id_hydro_only[t] for t in range(N)]
        day_df._batt_soc_end        = soc
    else:
        # No battery co-simulation — ID based on turbine vs forecast
        id_component = [
            -abs(opt_m2[t] + opt_m1[t] - forecast_list[t]) * TIMESTEP_HOURS * id_price[t] / 1000.0
            for t in range(N)]

    day_df['Opt_DA_Trading_EUR']     = da_component
    day_df['Opt_ID_Imbalance_EUR']   = id_component
    day_df['Opt_Energy_Trading_EUR'] = [da_component[t] + id_component[t] for t in range(N)]
    day_df['Forecast_Drift_kW']      = [
        (opt_m2[t] + opt_m1[t]) - forecast_list[t] for t in range(N)]
    day_df['Opt_Recovery_State']     = opt_state

    return day_df
