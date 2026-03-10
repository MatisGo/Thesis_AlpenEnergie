"""
Data Loading
============
Loads and cleans operational data from the Excel file.

Columns returned:
  DateTime       : timestamp (5-min resolution)
  Rain_mm        : precipitation (mm per 24h)
  Production_kW  : total measured turbine production (kW)
  Consumption_kW : local electricity demand (kW)
  Temperature    : outside temperature (°C)
  Irradiance     : solar irradiance (W/m²)
  Haselholz_mm   : Haselholz reservoir water level (mm)
  Bidmi_mm       : Bidmi reservoir water level (mm)
"""

import os
import pandas as pd
import numpy as np

# Path to the data file (one folder up from this file)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'matis_2025_.xlsx')


def load_data(start_date=None, end_date=None):
    """
    Load and clean operational data from the Excel file.

    Parameters
    ----------
    start_date : str or None
        Filter data from this date, e.g. '2025-06-01'
    end_date : str or None
        Filter data up to this date, e.g. '2025-06-07'

    Returns
    -------
    df : pd.DataFrame
        Clean time-series with all operational signals.
    """

    # -----------------------------------------------------------------------
    # Load the Excel file
    # The first 2 rows are metadata (units + signal names), so we skip them
    # -----------------------------------------------------------------------
    df = pd.read_excel(DATA_PATH, header=None, skiprows=2)

    # Assign meaningful column names
    df.columns = [
        'DateTime',         # Timestamp
        'Date',             # Date (duplicate, not used)
        'Daytime',          # Time of day (not used)
        'Rain_mm',          # Precipitation (mm/24h)
        'Production_kW',    # Total turbine production (kW)
        'Consumption_kW',   # Local electricity demand (kW)
        'Temperature',      # Outside temperature (°C)
        'Irradiance',       # Solar irradiance (W/m²)
        'Haselholz_mm',     # Haselholz reservoir level (mm)
        'Bidmi_mm',         # Bidmi reservoir level (mm)
    ]

    # -----------------------------------------------------------------------
    # Parse datetime and sort chronologically
    # -----------------------------------------------------------------------
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Convert all signal columns to numbers
    # -----------------------------------------------------------------------
    numeric_cols = [
        'Rain_mm', 'Production_kW', 'Consumption_kW',
        'Temperature', 'Irradiance', 'Haselholz_mm', 'Bidmi_mm'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values by linear interpolation (small gaps in the sensors)
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # -----------------------------------------------------------------------
    # Convert reservoir levels from metres (Excel) to millimetres (model units)
    # -----------------------------------------------------------------------
    df['Haselholz_mm'] = df['Haselholz_mm'] * 1000
    df['Bidmi_mm']     = df['Bidmi_mm']     * 1000

    # -----------------------------------------------------------------------
    # Optional: filter to a specific date range
    # -----------------------------------------------------------------------
    if start_date is not None:
        df = df[df['DateTime'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['DateTime'] <= pd.to_datetime(end_date)]

    df = df.reset_index(drop=True)

    print(f"  Loaded {len(df)} timesteps  "
          f"({df['DateTime'].iloc[0]} → {df['DateTime'].iloc[-1]})")

    return df
