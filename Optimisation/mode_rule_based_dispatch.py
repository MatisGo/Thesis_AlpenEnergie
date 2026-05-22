"""
mode_rule_based_dispatch.py
===========================
RULE_BASED_DISPATCH mode — rule-based dispatch that follows the CNN/LSTM forecast and
absorbs the consumption forecast error so the grid exchange stays equal to
the DA bid.

Turbine target
--------------
  turbine_target = Forecast_Production + (Consumption_real - Forecast_Consumption)

  When real consumption deviates from the predicted consumption, the turbines
  compensate so the net grid position stays at the DA bid
  (Forecast_Production - Forecast_Consumption). Reservoir constraints
  (RECOVERY / HALF_POWER) can override this target.

State machine
-------------
  NORMAL       Both turbines run, equalization split keeps both reservoirs at
               the same fill percentage. Reservoirs may sit anywhere between
               20% and 50% (or above 50%) in this mode.
               -> RECOVERY when any reservoir drops below 20%.

  RECOVERY     Both turbines OFF. Triggered when any reservoir drops below 20%.
               -> NORMAL    when BOTH reservoirs reach 50% on the same step.
               -> HALF_POWER when only ONE reservoir reaches 50%.

  HALF_POWER   The reservoir above 50% produces alone at 40% of turbine_target.
               - Bidmi   ≥50%  →  M2 alone (cascade also helps fill Haselholz)
               - Haselholz ≥50% →  M1 alone
               -> NORMAL   when the second reservoir also reaches 50%.
               -> RECOVERY when the active (producing) reservoir drops <20%.

  PEAK_SHAVING Kept as an optional override (PEAK_SHAVING_ENABLED flag) — not
               active by default.

Equalization formula (NORMAL state)
-----------------------------------
  Find M2 such that norm_lb_next == norm_lh_next, given M1 = target - M2.
  Analytical solution:
      A      = lb + inflow_b / BIDMI_LS_PER_MM - BIDMI_LEVEL_MIN
      B      = lh + inflow_h / HASELHOLZ_LS_PER_MM
               - COEFF_M1_HASELHOLZ × target - HASELHOLZ_LEVEL_MIN
      M2_eq  = (HASELHOLZ_RANGE × A - BIDMI_RANGE × B)
               / (BIDMI_RANGE × C2 + HASELHOLZ_RANGE × C1)
  where C1 = COEFF_M2_BIDMI, C2 = COEFF_M2_CASCADE + COEFF_M1_HASELHOLZ.

Pricing
-------
  DA component  = (Forecast_Production - Forecast_Consumption) × DA_price × dt / 1000  [EUR]
  ID component  = deviation × BG_price × dt / 1000  [EUR]
                  Over-delivery settled at BG_Long, under-delivery at BG_Short (AE tariff).
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
# Peak shaving toggle  —  set to True to re-enable the hard boost override
# ---------------------------------------------------------------------------
PEAK_SHAVING_ENABLED = True    # True → boost turbines in NORMAL/FILL_H to avoid new peak

# ---------------------------------------------------------------------------
# Recovery thresholds  [fraction of physical range]
# ---------------------------------------------------------------------------
RECOVERY_LOW_PCT  = 0.10   # any reservoir below this level → RECOVERY (turbines off)
RECOVERY_EXIT_PCT = 0.40   # reservoir reaching this level → exit RECOVERY
                            # (HALF_POWER if only one ≥50%, NORMAL if both)
HALF_POWER_FRAC   = 0.40   # HALF_POWER target = 0.40 × turbine_target

# Equalization coefficients
C1 = COEFF_M2_BIDMI                       # mm of Bidmi drop per kW of M2
C2 = COEFF_M2_CASCADE + COEFF_M1_HASELHOLZ  # mm of Haselholz rise per kW of M2


def dispatch_day(day_df, lb0, lh0, target_lb, target_lh, battery_cfg=None,
                 running_peak_kW=0.0, **_kwargs):
    """
    Rule-based dispatch for one day.

    battery_cfg    : BatteryConfig or None.
      If mode == 'HYDRO_COUPLED', the battery is co-simulated at each timestep:
        - battery_step_forecast() decides charge/discharge from current fill levels
        - turbine target adjusts so net grid export stays = Forecast
        - water balance uses the adjusted turbine output → levels are correct
      Results are written as Batt_* columns; _batt_soc_end is stored as an attribute.
      If mode == 'DAY_AHEAD' or None, battery is ignored here (handled separately).
    running_peak_kW: highest 15-min avg grid import already seen this calendar month [kW].
                     In NORMAL state the turbine target is boosted so that grid import
                     stays at or below this value (peak shaving, priority > forecast).
    """
    day_df        = day_df.reset_index(drop=True)
    N             = len(day_df)
    demand        = day_df['Consumption_kW'].tolist()
    forecast_cons = day_df['Forecast_Consumption_kW'].tolist()
    inflow_b      = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h      = day_df['Haselholz_Inflow_ls'].tolist()
    hours         = day_df['DateTime'].dt.hour.tolist()   # for peak-tariff window check

    co_sim = battery_cfg is not None and battery_cfg.mode in ('HYDRO_COUPLED', 'HYBRID')
    soc    = battery_cfg.soc0 if co_sim else 0.0

    opt_m2, opt_m1           = [], []
    opt_lb, opt_lh           = [], []
    opt_spill_b, opt_spill_h = [], []
    opt_state                = []
    opt_mode_num             = []   # 1=Normal 2=Recovery 3=Half_Power 4=Peak_Shaving
    batt_c_list, batt_d_list, batt_soc_list = [], [], []

    lb, lh = lb0, lh0

    # Initial state: RECOVERY if either reservoir is already below 20%, else NORMAL.
    # HALF_POWER is only ever entered as a stepping stone out of RECOVERY.
    norm_lb0 = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
    norm_lh0 = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE
    state = 'RECOVERY' if (norm_lb0 < RECOVERY_LOW_PCT or norm_lh0 < RECOVERY_LOW_PCT) else 'NORMAL'
    active_side = None   # set to 'BIDMI' or 'HASELHOLZ' when state == 'HALF_POWER'

    for t in range(N):
        forecast_kw = max(0.0, float(day_df.loc[t, 'Forecast_kW']))

        # Consumption forecast error correction: produce extra to absorb the gap
        # between real and predicted consumption, so grid exchange stays = DA bid.
        cons_error = demand[t] - forecast_cons[t]   # >0 when real demand exceeds forecast

        norm_lb = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
        norm_lh = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE

        # ── Battery co-simulation (HYDRO_COUPLED / HYBRID mode) ───────────────
        # Decision is based on reservoir fill at START of this timestep.
        # Turbine target shifts so that grid export remains = DA bid.
        # For HYBRID, soc_max_override caps the HYDRO_COUPLED zone below the DA
        # block floor so committed energy stays reserved.
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
        # NORMAL → RECOVERY: any reservoir below 20%
        # RECOVERY → NORMAL: BOTH reservoirs ≥50% on the same step
        # RECOVERY → HALF_POWER: only ONE reservoir ≥50%
        # HALF_POWER → NORMAL: second reservoir reaches 50%
        # HALF_POWER → RECOVERY: ACTIVE reservoir drops <20%
        if state == 'NORMAL':
            if norm_lb < RECOVERY_LOW_PCT or norm_lh < RECOVERY_LOW_PCT:
                state = 'RECOVERY'
                active_side = None
        elif state == 'RECOVERY':
            if norm_lb >= RECOVERY_EXIT_PCT and norm_lh >= RECOVERY_EXIT_PCT:
                state = 'NORMAL'
                active_side = None
            elif norm_lb >= RECOVERY_EXIT_PCT:
                state = 'HALF_POWER'
                active_side = 'BIDMI'
            elif norm_lh >= RECOVERY_EXIT_PCT:
                state = 'HALF_POWER'
                active_side = 'HASELHOLZ'
        elif state == 'HALF_POWER':
            if norm_lb >= RECOVERY_EXIT_PCT and norm_lh >= RECOVERY_EXIT_PCT:
                state = 'NORMAL'
                active_side = None
            elif (active_side == 'BIDMI'     and norm_lb < RECOVERY_LOW_PCT) or \
                 (active_side == 'HASELHOLZ' and norm_lh < RECOVERY_LOW_PCT):
                state = 'RECOVERY'
                active_side = None

        # ── Dispatch per state (no ramp applied here yet) ────────────────
        if state == 'RECOVERY':
            m2 = m1 = 0.0

        elif state == 'HALF_POWER':
            half_target = HALF_POWER_FRAC * turbine_target
            if active_side == 'BIDMI':
                m2 = max(0.0, min(P_MAX_M2, half_target))
                m1 = 0.0
            else:  # HASELHOLZ
                m1 = max(0.0, min(P_MAX_M1, half_target))
                m2 = 0.0

        else:  # NORMAL — equalization split targeting turbine_target
            A = lb + inflow_b[t] / BIDMI_LS_PER_MM - BIDMI_LEVEL_MIN
            B = (lh + inflow_h[t] / HASELHOLZ_LS_PER_MM
                 - COEFF_M1_HASELHOLZ * turbine_target - HASELHOLZ_LEVEL_MIN)
            m2_eq = (HASELHOLZ_RANGE * A - BIDMI_RANGE * B) / (BIDMI_RANGE * C2 + HASELHOLZ_RANGE * C1)
            m2 = max(0.0, min(P_MAX_M2, m2_eq, turbine_target))
            m1_target = turbine_target - m2
            m1 = max(0.0, min(P_MAX_M1, m1_target))

        # ── Total-sum ramp constraint  (200 kW per 15 min on M1+M2 combined) ──
        # Physical plant limit: the two turbines together cannot change output
        # by more than RAMP_MAX in a RAMP_WINDOW (15-min) span. We compare
        # (m2+m1) against (opt_m2[t-W] + opt_m1[t-W]) and scale both turbines
        # proportionally if the change would exceed RAMP_MAX. This applies
        # even in RECOVERY (gradual shutdown — physically realistic).
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
                    # cur_total = 0 (e.g. RECOVERY) but ramp says drop more slowly
                    # → keep previous turbine ratio
                    m2 = new_total * (opt_m2[t - RAMP_WINDOW] / prev_total)
                    m1 = new_total * (opt_m1[t - RAMP_WINDOW] / prev_total)

        # ── Peak shaving — overrides ALL states (including RECOVERY) ─────
        # Toggle with PEAK_SHAVING_ENABLED at the top of this file.
        # Active only during the peak-tariff window (PEAK_TARIFF_HOUR_*).
        # M1 is boosted first (drains Haselholz, not Bidmi).
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

        # Numeric dispatch mode: 1=Normal 2=Recovery 3=Half_Power 4=Peak_Shaving
        if peak_active:
            mode_num = 4
        elif state == 'RECOVERY':
            mode_num = 2
        elif state == 'HALF_POWER':
            mode_num = 3
        else:
            mode_num = 1

        # Water balance uses ACTUAL turbine output — levels are always correct
        lb, lh, spill_b, spill_h, m2, m1 = water_balance_step(lb, lh, m2, m1, inflow_b[t], inflow_h[t])

        opt_m2.append(m2);  opt_m1.append(m1)
        opt_lb.append(lb);  opt_lh.append(lh)
        opt_spill_b.append(spill_b); opt_spill_h.append(spill_h)
        opt_state.append(state)
        opt_mode_num.append(mode_num)

    da_price      = day_df['Day_Ahead_Price_EUR_MWh'].tolist()
    bg_long_r     = day_df['BG_Long_EUR_MWh'].tolist()   # real prices for settlement
    bg_short_r    = day_df['BG_Short_EUR_MWh'].tolist()
    forecast_list = [max(0.0, float(day_df.loc[t, 'Forecast_kW'])) for t in range(N)]

    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, target_lb, target_lh,
        mode_name='RULE_BASED_DISPATCH',
    )

    # DA component: bid = Forecast_Production - Forecast_Consumption (net position, no look-ahead)
    da_bid_list  = [forecast_list[t] - forecast_cons[t] for t in range(N)]
    da_component = [
        da_bid_list[t] * TIMESTEP_HOURS * da_price[t] / 1000.0
        for t in range(N)]

    if co_sim:
        # ID imbalance: (actual net grid position) - DA bid
        # actual net = turbine + battery_net - consumption
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

        # Battery revenue = ID improvement vs hydro-only
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
        # No battery — ID on (turbine - consumption) vs DA bid
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
        (opt_m2[t] + opt_m1[t]) - forecast_list[t] for t in range(N)]
    day_df['Opt_Recovery_State']     = opt_state
    day_df['Opt_Dispatch_Mode']      = opt_mode_num
    # 1=Normal  2=Recovery  3=Half_Power  4=Peak_Shaving

    return day_df
