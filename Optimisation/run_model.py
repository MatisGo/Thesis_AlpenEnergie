"""
Simple data prep script
=======================

Reads the source Excel file and prepares a clean DataFrame with:
  - reservoir levels in mm (Bidmi / Haselholz)
  - a per-turbine production split based on a fixed ratio

Usage:
  python run_model.py

The only external dependency is pandas (for reading Excel).
"""

import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Ratio MII/MI (Haselholz / Bidmi). This is the ratio of the two turbines.
# With this ratio, we split the measured total production into the two turbine
# contributions (Bidmi = MI, Haselholz = MII).
#
# Example:
#   total_prod = MI + MII
#   MII = ratio * MI
#   => MI = total_prod / (1 + ratio)
#
TURBINE_RATIO_MII_MI = 1.560

# Water-to-energy conversion factors (kWh per mm of reservoir drop)
BIDMI_KWH_PER_MM = 3.33
HASELHOLZ_KWH_PER_MM = 0.455

# Timestep duration in hours (5-minute resolution)
TIMESTEP_HOURS = 5 / 60


DATA_FILENAME = 'matis_2025_.xlsx'
OUTPUT_FILENAME = 'results.csv'


def load_data(path):
    """Load the Excel file and clean the data."""

    # The source data has three header rows (names / units / signal names), skip all.
    df = pd.read_excel(path, header=None, skiprows=3)

    # Column layout in the Excel file
    df.columns = [
        'DateTime',
        'Date',
        'Daytime',
        'Rain_mm',
        'Production_kW',
        'Consumption_kW',
        'Temperature',
        'Irradiance',
        'Haselholz_m',
        'Bidmi_m',
        'Spot_Price_CHF_MWh',
    ]

    # Parse datetime — source format is DD.MM.YYYY HH:MM:SS (European), so
    # dayfirst=True is required to avoid month/day swaps for dates where both
    # day and month are <= 12.
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = [
        'Rain_mm',
        'Production_kW',
        'Consumption_kW',
        'Temperature',
        'Irradiance',
        'Haselholz_m',
        'Bidmi_m',
        'Spot_Price_CHF_MWh',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Interpolate missing values
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # Convert reservoir levels from metres to millimetres
    df['Haselholz_mm'] = df['Haselholz_m'] * 1000
    df['Bidmi_mm'] = df['Bidmi_m'] * 1000

    # Compute per-turbine production given the fixed ratio
    ratio = TURBINE_RATIO_MII_MI
    df['Bidmi_Production_kW'] = df['Production_kW'] / (1 + ratio)
    df['Haselholz_Production_kW'] = df['Bidmi_Production_kW'] * ratio

    # Convert production (kW) to turbine water output (mm) per timestep
    # kW * timestep_h = kWh consumed; kWh / (kWh/mm) = mm of water used
    df['Bidmi_Output_mm'] = (df['Bidmi_Production_kW'] * TIMESTEP_HOURS) / BIDMI_KWH_PER_MM
    df['Haselholz_Output_mm'] = (df['Haselholz_Production_kW'] * TIMESTEP_HOURS) / HASELHOLZ_KWH_PER_MM

    # Water inflow per timestep (mm)
    # inflow = (level(t-1) - level(t)) - turbine_output(t)
    # i.e. unexplained level drop beyond turbine usage = net inflow received
    df['Bidmi_Inflow_mm'] = df['Bidmi_mm'].shift(1) - df['Bidmi_mm'] - df['Bidmi_Output_mm']
    df['Haselholz_Inflow_mm'] = df['Haselholz_mm'].shift(1) - df['Haselholz_mm'] - df['Haselholz_Output_mm']

    # Natural inflow parameters for the optimisation model (physically correct signs)
    # Derived from the water balance equation:
    #   Level(t) = Level(t-1) + NaturalInflow(t) [+ cascade] - Discharge(t)
    # => NaturalInflow(t) = ΔLevel(t) + Discharge(t) [- cascade_in]
    #
    # Bidmi (no cascade input):
    #   NatInflow_B(t) = (Level_B(t) - Level_B(t-1)) + Discharge_B(t)
    df['Natural_Inflow_B_mm'] = df['Bidmi_mm'].diff() + df['Bidmi_Output_mm']
    #
    # Haselholz (receives Bidmi discharge as cascade input):
    #   Level_H(t) = Level_H(t-1) + NatInflow_H(t) + Discharge_B(t) - Discharge_H(t)
    #   => NatInflow_H(t) = ΔLevel_H(t) - Discharge_B(t) + Discharge_H(t)
    df['Natural_Inflow_H_mm'] = (
        df['Haselholz_mm'].diff()
        - df['Bidmi_Output_mm']
        + df['Haselholz_Output_mm']
    )
    # First row has no previous value — assume zero inflow for that step
    df['Natural_Inflow_B_mm'] = df['Natural_Inflow_B_mm'].fillna(0.0)
    df['Natural_Inflow_H_mm'] = df['Natural_Inflow_H_mm'].fillna(0.0)

    # External network exchange (kW): positive = importing from grid, negative = exporting
    df['Network_Exchange_kW'] = df['Consumption_kW'] - df['Production_kW']

    # Energy trading cost/revenue per timestep (CHF)
    # kW -> kWh: multiply by timestep in hours
    # kWh * CHF/MWh / 1000 kWh/MWh = CHF
    # Positive = cost (buying), negative = revenue (selling)
    df['Energy_Trading_CHF'] = (
        df['Network_Exchange_kW'] * TIMESTEP_HOURS * df['Spot_Price_CHF_MWh'] / 1000
    )

    # Cumulative total energy cost over the entire dataset (CHF)
    df['Total_Energy_Cost_CHF'] = df['Energy_Trading_CHF'].cumsum()

    return df


def main():
    root = os.path.dirname(__file__)
    source_path = os.path.join(root, DATA_FILENAME)

    if not os.path.exists(source_path):
        print(f"ERROR: could not find data file: {source_path}")
        sys.exit(1)

    print(f"Loading data from: {source_path}")
    df = load_data(source_path)

    print(f"Loaded {len(df)} rows (from {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[-1]})")
    print(f"Writing results to: {OUTPUT_FILENAME}")

    df.to_csv(OUTPUT_FILENAME, index=False)

    print("Done.")


if __name__ == '__main__':
    main()
