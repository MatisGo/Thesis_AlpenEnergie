"""
mode_water_level.py
===================
WATER_LEVEL mode — both turbines off.

Both M2 and M1 are set to zero.  The reservoirs recover entirely from
natural inflow.  Use this mode when both reservoirs are critically low
and need to be refilled before any production resumes.
"""

from hydro_constants import (
    P_MAX_M2, P_MAX_M1,
    water_balance_step, attach_common_results,
)


def dispatch_day(day_df, lb0, lh0, target_lb, target_lh, **_kwargs):
    """
    Both turbines off for the entire day.
    Reservoirs fill from natural inflow only.
    """
    day_df   = day_df.reset_index(drop=True)
    N        = len(day_df)
    demand   = day_df['Consumption_kW'].tolist()
    price    = day_df['Spot_Price_CHF_MWh'].tolist()
    inflow_b = day_df['Bidmi_Inflow_ls'].tolist()
    inflow_h = day_df['Haselholz_Inflow_ls'].tolist()

    opt_m2, opt_m1       = [], []
    opt_lb, opt_lh       = [], []
    opt_spill_b, opt_spill_h = [], []

    lb, lh = lb0, lh0

    for t in range(N):
        m2, m1 = 0.0, 0.0
        lb, lh, spill_b, spill_h = water_balance_step(lb, lh, m2, m1, inflow_b[t], inflow_h[t])

        opt_m2.append(m2);      opt_m1.append(m1)
        opt_lb.append(lb);      opt_lh.append(lh)
        opt_spill_b.append(spill_b); opt_spill_h.append(spill_h)

    return attach_common_results(
        day_df, opt_m2, opt_m1, opt_lb, opt_lh,
        opt_spill_b, opt_spill_h,
        demand, price, target_lb, target_lh,
        mode_name='WATER_LEVEL',
    )
