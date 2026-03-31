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

# ---------------------------------------------------------------------------
# Recovery thresholds  [fraction of physical range]
# ---------------------------------------------------------------------------
RECOVERY_LOW_PCT  = 0.20   # enter recovery below this level
RECOVERY_EXIT_PCT = 0.50   # Bidmi must reach this to start filling Haselholz
                            # Haselholz must reach this to resume normal mode

# Equalization coefficients
C1 = COEFF_M2_BIDMI                       # mm of Bidmi drop per kW of M2
C2 = COEFF_M2_CASCADE + COEFF_M1_HASELHOLZ  # mm of Haselholz rise per kW of M2


def dispatch_day(day_df, lb0, lh0, target_lb, target_lh, **_kwargs):
    """
    Rule-based dispatch for one day.
    Returns day_df with Opt_* columns and Forecast_Drift_kW added.
    """
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    demand   = day_df['Consumption_kW'].tolist()
    price    = day_df['Spot_Price_CHF_MWh'].tolist()
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    opt_m2, opt_m1           = [], []
    opt_lb, opt_lh           = [], []
    opt_spill_b, opt_spill_h = [], []
    opt_state                = []

    lb, lh = lb0, lh0

    # Initialise recovery state
    norm_lb0 = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
    norm_lh0 = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE
    state = 'RECOVERY' if (norm_lb0 < RECOVERY_LOW_PCT or norm_lh0 < RECOVERY_LOW_PCT) else 'NORMAL'

    for t in range(N):
        forecast_kw = max(0.0, float(day_df.loc[t, 'Forecast_kW']))

        norm_lb = (lb - BIDMI_LEVEL_MIN)     / BIDMI_RANGE
        norm_lh = (lh - HASELHOLZ_LEVEL_MIN) / HASELHOLZ_RANGE

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

        else:  # NORMAL — equalization split
            A = lb + inflow_b[t] / BIDMI_LS_PER_MM - BIDMI_LEVEL_MIN
            B = (lh + inflow_h[t] / HASELHOLZ_LS_PER_MM
                 - COEFF_M1_HASELHOLZ * forecast_kw - HASELHOLZ_LEVEL_MIN)
            m2_eq = (HASELHOLZ_RANGE * A - BIDMI_RANGE * B) / (BIDMI_RANGE * C2 + HASELHOLZ_RANGE * C1)

            m2 = max(0.0, min(P_MAX_M2, m2_eq, forecast_kw))
            if t >= RAMP_WINDOW:
                m2 = max(opt_m2[t - RAMP_WINDOW] - RAMP_MAX,
                         min(opt_m2[t - RAMP_WINDOW] + RAMP_MAX, m2))

            m1_target = forecast_kw - m2
            m1 = max(0.0, min(P_MAX_M1, m1_target))
            if t >= RAMP_WINDOW:
                m1 = max(opt_m1[t - RAMP_WINDOW] - RAMP_MAX,
                         min(opt_m1[t - RAMP_WINDOW] + RAMP_MAX, m1))

        lb, lh, spill_b, spill_h = water_balance_step(lb, lh, m2, m1, inflow_b[t], inflow_h[t])

        opt_m2.append(m2);  opt_m1.append(m1)
        opt_lb.append(lb);  opt_lh.append(lh)
        opt_spill_b.append(spill_b); opt_spill_h.append(spill_h)
        opt_state.append(state)

    attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, price, target_lb, target_lh,
        mode_name='FORECAST',
    )

    # Forecast-specific extra columns
    day_df['Forecast_Drift_kW'] = [
        (opt_m2[t] + opt_m1[t]) - float(day_df.loc[t, 'Forecast_kW'])
        for t in range(N)]
    day_df['Opt_Recovery_State'] = opt_state

    return day_df
