"""
Parameters
==========
All fixed inputs to the optimisation model.

This file has two sections:
  A) Physical system constants  → turbine factors, reservoir limits, etc.
  B) Time-series parameters     → demand, inflow, prices (built from data)

⚠️  Parameters marked with PLACEHOLDER need to be updated
    with real measured values from the system documentation.
"""

import numpy as np

# ===========================================================================
# A) PHYSICAL SYSTEM CONSTANTS
# ===========================================================================

# --- Turbine conversion factors ---
# How many kWh of electricity are produced per mm of water discharged
# (over one 5-minute timestep)
K_TURBINE = {
    'M1': 0.455,    # Haselholz turbine  [kWh / mm]
    'M2': 3.330,    # Bidmi turbine       [kWh / mm]
}

# --- Maximum water discharge per turbine per 5-min timestep ---
# ⚠️  Check this values with the real turbine power
QMAX_TURBINE = {
    'M1': 241.60,      # [mm per 5-min step]
    'M2': 46.25,      # [mm per 5-min step]
}

# --- Reservoir storage limits ---
# Maximum and minimum water levels in the reservoirs (in mm), which determine
RMAX_RESERVOIR = {
    'Bidmi':      2200.0,     # Maximum safe water level [mm]
    'Haselholz':  2800.0,     # Maximum safe water level [mm]
}

RMIN_RESERVOIR = {
    'Bidmi':      1000.0,      # Minimum operating water level [mm]
    'Haselholz':  600.0,      # Minimum operating water level [mm]
}

# --- Winter energy contract ---
# Minimum electricity production commitment during winter months [kWh/step]
# ⚠️  PLACEHOLDER — set from the actual winter contract
CONTRACT_KWH = 0.0

# Winter months (1=Jan, 2=Feb, ..., 12=Dec)
WINTER_MONTHS = [11, 12, 1, 2, 3]

# --- Timestep duration ---
# One timestep = 5 minutes = 5/60 hours
DELTA_T = 5 / 60   # [hours]


# ===========================================================================
# B) TIME-SERIES PARAMETERS  (built from data)
# ===========================================================================

def get_parameters(df, sets):
    """
    Build all parameter dictionaries needed by the model.

    Parameters
    ----------
    df   : pd.DataFrame   output of load_data()
    sets : dict           output of get_sets()

    Returns
    -------
    params : dict
        All parameters, indexed by (t) or (r, t) or (i).
    """

    T = sets['T']

    # -----------------------------------------------------------------------
    # 1. DEMAND  [kWh per 5-min timestep]
    # -----------------------------------------------------------------------
    # The Excel has power in kW. We convert to energy:
    #   Energy [kWh] = Power [kW] × Duration [h] = kW × (5/60)
    Demand = {
        t: float(df['Consumption_kW'].iloc[t]) * DELTA_T
        for t in T
    }

    # -----------------------------------------------------------------------
    # 2. NATURAL INFLOW TO EACH RESERVOIR  [mm per 5-min timestep]
    # -----------------------------------------------------------------------
    # We derive the implied inflow from the observed reservoir level changes.
    #
    # Reservoir balance:
    #   R[t+1] = R[t] + Inflow[t] - Discharge[t]
    #   → Inflow[t] = R[t+1] - R[t] + Discharge[t]
    #
    # Since we don't have the discharge split per turbine in the raw data,
    # we use the observed level INCREASE as a lower-bound proxy for inflow.
    # Negative changes (level drops) are set to zero here because they
    # include the turbine discharge effect.
    #
    # ⚠️  Calibrate this with real flow measurements if available.

    bidmi_levels    = df['Bidmi_mm'].values
    haselholz_levels = df['Haselholz_mm'].values

    # Change in level between consecutive timesteps
    delta_bidmi    = np.diff(bidmi_levels,    append=bidmi_levels[-1])
    delta_haselholz = np.diff(haselholz_levels, append=haselholz_levels[-1])

    Inflow = {}
    for t in T:
        # Only positive changes indicate net inflow (rain/snowmelt)
        Inflow['Bidmi',     t] = max(0.0, float(delta_bidmi[t]))
        Inflow['Haselholz', t] = max(0.0, float(delta_haselholz[t]))

    # -----------------------------------------------------------------------
    # 3. ELECTRICITY MARKET PRICES  [€ / kWh]
    # -----------------------------------------------------------------------
    # ⚠️  PLACEHOLDER — TODO later: use real price data from the market for each timestep.
    # Real prices can be downloaded from ENTSO-E Transparency Platform.
    #
    # For now: flat price of 10 ct/kWh for spot, 12 ct/kWh for intraday.

    Price_spot     = {t: 0.10 for t in T}   # 10 ct/kWh
    Price_intraday = {t: 0.12 for t in T}   # 12 ct/kWh

    # -----------------------------------------------------------------------
    # 4. WINTER CONTRACT  [kWh per timestep]
    # -----------------------------------------------------------------------
    is_winter = df['DateTime'].dt.month.isin(WINTER_MONTHS)
    Contract = {
        t: CONTRACT_KWH if is_winter.iloc[t] else 0.0
        for t in T
    }

    # -----------------------------------------------------------------------
    # 5. INITIAL RESERVOIR LEVELS  [mm]
    # -----------------------------------------------------------------------
    # The model starts from the first observed water level
    R0 = {
        'Bidmi':     float(df['Bidmi_mm'].iloc[0]),
        'Haselholz': float(df['Haselholz_mm'].iloc[0]),
    }

    # -----------------------------------------------------------------------
    # Pack everything into one dictionary
    # -----------------------------------------------------------------------
    params = {
        'K_turbine':      K_TURBINE,
        'Qmax':           QMAX_TURBINE,
        'Rmax':           RMAX_RESERVOIR,
        'Rmin':           RMIN_RESERVOIR,
        'Demand':         Demand,
        'Inflow':         Inflow,
        'Price_spot':     Price_spot,
        'Price_intraday': Price_intraday,
        'Contract':       Contract,
        'R0':             R0,
        'delta_t':        DELTA_T,
    }

    print(f"  Parameters built for {len(T)} timesteps")
    print(f"  Initial Bidmi level:     {R0['Bidmi']:.3f} mm")
    print(f"  Initial Haselholz level: {R0['Haselholz']:.3f} mm")
    print(f"  Total demand:            {sum(Demand.values()):.1f} kWh")

    return params
