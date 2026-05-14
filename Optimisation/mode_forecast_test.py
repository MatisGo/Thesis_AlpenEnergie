"""
mode_forecast_test.py
=====================
FORECAST_TEST — experimental rule-based dispatch matching the state-machine
diagram (RECOVERY → FILL_H → NORMAL).

Differences vs mode_forecast.py:
  - FILL_H replaces HALF_POWER. M2 runs at full P_MAX_M2 (1920 kW), M1 = 0.
  - From RECOVERY, if Haselholz reaches 50% first, go directly to NORMAL
    (skip FILL_H). FILL_H is only entered if Bidmi reaches 50% first.
  - Peak shaving (capacity-tariff floor) overrides ALL states, including
    RECOVERY. M1 is boosted first, then M2 if needed.
  - Consumption forecast error correction is kept:
        turbine_target = Forecast + (Consumption_real − Forecast_Consumption)

State machine
-------------
  NORMAL       Equalization split, M2 + M1 follow turbine_target.
  RECOVERY     Both turbines off. Priority override: any reservoir < 20%.
  FILL_H       M2 = 1920 kW, M1 = 0. Cascade fills Haselholz.
  PEAK_SHAVING Boost (M1 first, then M2) on top of any state when grid
               import would set a new monthly peak.

Transitions
-----------
  any state    → RECOVERY     when norm_lb < 20% or norm_lh < 20%
  RECOVERY     → NORMAL       when norm_lh ≥ 50% (regardless of Bidmi)
  RECOVERY     → FILL_H       when norm_lb ≥ 50% and norm_lh < 50%
  FILL_H       → NORMAL       when norm_lh ≥ 50%

Pricing
-------
  DA component  = (Forecast_Production - Forecast_Consumption) × DA_price × dt / 1000  [EUR]
  ID component  = deviation × BG_price × dt / 1000  [EUR]
                  Over-delivery → BG_Long, under-delivery → BG_Short (AE tariff).
"""

import pandas as pd
from hydro_constants import (
    P_MAX_M2, P_MAX_M1, RAMP_MAX, RAMP_WINDOW, TIMESTEP_HOURS,
    BIDMI_LEVEL_MIN, HASELHOLZ_LEVEL_MIN,
    BIDMI_RANGE, HASELHOLZ_RANGE,
    COEFF_M2_BIDMI, COEFF_M2_CASCADE, COEFF_M1_HASELHOLZ,
    BIDMI_LS_PER_MM, HASELHOLZ_LS_PER_MM,
    PEAK_TARIFF_HOUR_START, PEAK_TARIFF_HOUR_END,
    water_balance_step, attach_common_results,
)
from battery_control import battery_step_forecast

# ---------------------------------------------------------------------------
# Peak shaving — always on in this test mode, overrides all states (incl. RECOVERY)
# ---------------------------------------------------------------------------
PEAK_SHAVING_ENABLED = True

# ---------------------------------------------------------------------------
# Recovery thresholds  [fraction of physical range]
# ---------------------------------------------------------------------------
RECOVERY_LOW_PCT  = 0.20   # any reservoir below this level → RECOVERY
RECOVERY_EXIT_PCT = 0.50   # reservoir reaching this level → exit RECOVERY

# Equalization coefficients
C1 = COEFF_M2_BIDMI                       # mm of Bidmi drop per kW of M2
C2 = COEFF_M2_CASCADE + COEFF_M1_HASELHOLZ  # mm of Haselholz rise per kW of M2


def dispatch_day(day_df, lb0, lh0, target_lb, target_lh, battery_cfg=None,
                 running_peak_kW=0.0, **_kwargs):
    """Rule-based dispatch for one day (FORECAST_TEST flavour)."""
    day_df        = day_df.reset_index(drop=True)
    N             = len(day_df)
    demand        = day_df['Consumption_kW'].tolist()
    forecast_cons = day_df['Forecast_Consumption_kW'].tolist()
    inflow_b      = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h      = day_df['Haselholz_Inflow_ls'].tolist()
    hours         = day_df['DateTime'].dt.hour.tolist()   # for peak-tariff window check

    co_sim = battery_cfg is not None and battery_cfg.mode in ('INTRADAY', 'HYBRID')
    soc    = battery_cfg.soc0 if co_sim else 0.0

    opt_m2, opt_m1           = [], []
    opt_lb, opt_lh           = [], []
    opt_spill_b, opt_spill_h = [], []
    opt_state                = []
    opt_mode_num             = []   # 1=Normal 2=Recovery 3=Fill_H 4=Peak_Shaving
    batt_c_list, batt_d_list, batt_soc_list = [], [], []

    lb, lh = lb0, lh0

    norm_lb0 = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
    norm_lh0 = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE
    state = 'RECOVERY' if (norm_lb0 < RECOVERY_LOW_PCT or norm_lh0 < RECOVERY_LOW_PCT) else 'NORMAL'

    for t in range(N):
        forecast_kw = max(0.0, float(day_df.loc[t, 'Forecast_kW']))

        # Consumption forecast error correction
        cons_error = demand[t] - forecast_cons[t]

        norm_lb = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
        norm_lh = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE

        # Battery co-simulation
        if co_sim:
            batt_soc_list.append(soc)
            batt_c, batt_d, soc = battery_step_forecast(soc, norm_lb, norm_lh)
            turbine_target = forecast_kw + cons_error + batt_c - batt_d
        else:
            batt_c = batt_d = 0.0
            turbine_target = forecast_kw + cons_error
        batt_c_list.append(batt_c)
        batt_d_list.append(batt_d)

        # ── State transitions ─────────────────────────────────────────────
        # Priority override: any reservoir < 20% → RECOVERY (from any state)
        if norm_lb < RECOVERY_LOW_PCT or norm_lh < RECOVERY_LOW_PCT:
            state = 'RECOVERY'
        elif state == 'RECOVERY':
            if norm_lh >= RECOVERY_EXIT_PCT:
                state = 'NORMAL'                   # skip FILL_H
            elif norm_lb >= RECOVERY_EXIT_PCT:
                state = 'FILL_H'
        elif state == 'FILL_H':
            if norm_lh >= RECOVERY_EXIT_PCT:
                state = 'NORMAL'

        # ── Dispatch per state (no ramp here — applied to sum below) ─────
        if state == 'RECOVERY':
            m2 = m1 = 0.0

        elif state == 'FILL_H':
            m2 = P_MAX_M2
            m1 = 0.0

        else:  # NORMAL — equalization split targeting turbine_target
            A = lb + inflow_b[t] / BIDMI_LS_PER_MM - BIDMI_LEVEL_MIN
            B = (lh + inflow_h[t] / HASELHOLZ_LS_PER_MM
                 - COEFF_M1_HASELHOLZ * turbine_target - HASELHOLZ_LEVEL_MIN)
            m2_eq = (HASELHOLZ_RANGE * A - BIDMI_RANGE * B) / (BIDMI_RANGE * C2 + HASELHOLZ_RANGE * C1)
            m2 = max(0.0, min(P_MAX_M2, m2_eq, turbine_target))
            m1_target = turbine_target - m2
            m1 = max(0.0, min(P_MAX_M1, m1_target))

        # ── Total-sum ramp constraint  (200 kW per 15 min on M1+M2 combined) ──
        if t >= RAMP_WINDOW:
            prev_total = opt_m2[t - RAMP_WINDOW] + opt_m1[t - RAMP_WINDOW]
            cur_total  = m2 + m1
            if cur_total > prev_total + RAMP_MAX:
                new_total = prev_total + RAMP_MAX
                if cur_total > 1e-9:
                    scale = new_total / cur_total
                    m2 *= scale; m1 *= scale
            elif cur_total < prev_total - RAMP_MAX:
                new_total = max(0.0, prev_total - RAMP_MAX)
                if cur_total > 1e-9:
                    scale = new_total / cur_total
                    m2 *= scale; m1 *= scale
                elif prev_total > 1e-9:
                    m2 = new_total * (opt_m2[t - RAMP_WINDOW] / prev_total)
                    m1 = new_total * (opt_m1[t - RAMP_WINDOW] / prev_total)

        # ── Peak shaving — overrides ALL states (including RECOVERY) ──────
        # Active only during the peak-tariff window (PEAK_TARIFF_HOUR_*).
        in_peak_window = PEAK_TARIFF_HOUR_START <= hours[t] < PEAK_TARIFF_HOUR_END
        peak_floor     = demand[t] - (batt_d - batt_c) - running_peak_kW
        peak_active    = PEAK_SHAVING_ENABLED and in_peak_window and peak_floor > m2 + m1
        if peak_active:
            extra  = peak_floor - (m2 + m1)
            m1_add = min(extra, max(0.0, P_MAX_M1 - m1))
            m1    += m1_add
            extra -= m1_add
            if extra > 0:
                m2 = min(P_MAX_M2, m2 + extra)

        # Mode number for output
        if peak_active:
            mode_num = 4
        elif state == 'RECOVERY':
            mode_num = 2
        elif state == 'FILL_H':
            mode_num = 3
        else:
            mode_num = 1

        lb, lh, spill_b, spill_h, m2, m1 = water_balance_step(lb, lh, m2, m1, inflow_b[t], inflow_h[t])

        opt_m2.append(m2);  opt_m1.append(m1)
        opt_lb.append(lb);  opt_lh.append(lh)
        opt_spill_b.append(spill_b); opt_spill_h.append(spill_h)
        opt_state.append(state)
        opt_mode_num.append(mode_num)

    da_price      = day_df['Day_Ahead_Price_EUR_MWh'].tolist()
    bg_long_r     = day_df['BG_Long_EUR_MWh'].tolist()
    bg_short_r    = day_df['BG_Short_EUR_MWh'].tolist()
    forecast_list = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]

    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, target_lb, target_lh,
        mode_name='FORECAST_TEST',
    )

    da_bid_list  = [forecast_list[t] - forecast_cons[t] for t in range(N)]
    da_component = [
        da_bid_list[t] * TIMESTEP_HOURS * da_price[t] / 1000.0
        for t in range(N)]

    if co_sim:
        grid_out  = [opt_m2[t] + opt_m1[t] + batt_d_list[t] - batt_c_list[t]
                     for t in range(N)]
        dev_total = [(grid_out[t] - demand[t]) - da_bid_list[t] for t in range(N)]
        id_component = [
            dev_total[t] * (bg_long_r[t] if dev_total[t] >= 0 else bg_short_r[t]) * TIMESTEP_HOURS / 1000.0
            for t in range(N)]
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
        id_component = [
            dev[t] * (bg_long_r[t] if dev[t] >= 0 else bg_short_r[t]) * TIMESTEP_HOURS / 1000.0
            for t in range(N)]

    day_df['Opt_DA_Trading_EUR']     = da_component
    day_df['Opt_ID_Imbalance_EUR']   = id_component
    day_df['Opt_Energy_Trading_EUR'] = [da_component[t] + id_component[t] for t in range(N)]
    day_df['Forecast_Drift_kW']      = [
        (opt_m2[t] + opt_m1[t]) - forecast_list[t] for t in range(N)]
    day_df['Opt_Recovery_State']     = opt_state
    day_df['Opt_Dispatch_Mode']      = opt_mode_num
    # 1=Normal  2=Recovery  3=Fill_H  4=Peak_Shaving

    return day_df