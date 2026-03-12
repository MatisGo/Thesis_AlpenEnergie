"""
get_weather_data.py
====================
Fetches weather data from Open-Meteo (free, no API key) for Meiringen,
Switzerland and interpolates to 5-minute resolution.

Two API calls are combined:
  1. Archive API  : historical data from START_DATE -> today
  2. Forecast API : next 72h forecast, ending at midnight 3 days from now

Location  : Meiringen (lat=46.7286, lon=8.1750)
Variables : temperature_2m (C), shortwave_radiation (W/m2), rain_sum (mm/day)
Output    : Test_Data_Import.xlsx  (Time | Date | Time_Only | ...)

Rain columns:
  Rain_Sum_mm          : historical daily rain total  (same value for all 5-min of the day)
  Rain_Forecast_Sum_mm : forecast  daily rain total   (same value for all 5-min of the day)

Open-Meteo docs:
  https://open-meteo.com/en/docs/historical-weather-api
  https://open-meteo.com/en/docs
"""

import os
import datetime

import requests
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =============================================================================
# CONFIGURATION
# =============================================================================

LAT        = 46.7286
LON        = 8.1750
START_DATE = "2025-01-01"
TIMEZONE   = "Europe/Zurich"

TODAY        = datetime.date.today()
FORECAST_END = datetime.datetime.combine(TODAY + datetime.timedelta(days=4),
                                         datetime.time(0, 0, 0))  # midnight of day+4

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "Imported_Forecast.xlsx")

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# =============================================================================
# FETCH HELPERS
# =============================================================================

def _parse_response(resp: requests.Response) -> tuple[pd.DataFrame, dict]:
    """
    Parse an Open-Meteo JSON response.

    Returns
    -------
    df_hourly  : DataFrame [DateTime, Temperature_C, Irradiance_Wm2]
    rain_map   : dict {datetime.date -> rain_sum_mm}  (from daily data)
    """
    if resp.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo returned HTTP {resp.status_code}:\n{resp.text}")

    payload = resp.json()

    # --- Hourly ---
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError("Unexpected response format from Open-Meteo.")

    df = pd.DataFrame({
        "DateTime":       pd.to_datetime(hourly["time"]),
        "Temperature_C":  pd.to_numeric(hourly["temperature_2m"],     errors="coerce"),
        "Irradiance_Wm2": pd.to_numeric(hourly["shortwave_radiation"], errors="coerce"),
    })
    df["Temperature_C"]  = df["Temperature_C"].interpolate(method="linear")
    df["Irradiance_Wm2"] = df["Irradiance_Wm2"].interpolate(method="linear").clip(lower=0)

    # --- Daily rain sum -> {date: mm} ---
    daily    = payload.get("daily", {})
    rain_map = {}
    if "time" in daily and "rain_sum" in daily:
        for t, r in zip(daily["time"], daily["rain_sum"]):
            d = pd.to_datetime(t).date()
            rain_map[d] = float(r) if r is not None else 0.0

    return df, rain_map


def fetch_historical() -> tuple[pd.DataFrame, dict]:
    """Fetch historical hourly data + daily rain from Open-Meteo archive."""
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": START_DATE,
        "end_date":   TODAY.strftime("%Y-%m-%d"),
        "hourly":     "temperature_2m,shortwave_radiation",
        "daily":      "rain_sum",
        "timezone":   TIMEZONE,
    }
    print("  [1/2] Calling Open-Meteo archive API (historical) ...")
    df, rain_map = _parse_response(requests.get(ARCHIVE_URL, params=params, timeout=60))
    print(f"        {len(df):,} hourly records  "
          f"({df['DateTime'].iloc[0]}  to  {df['DateTime'].iloc[-1]})")
    print(f"        {len(rain_map)} daily rain_sum values")
    return df, rain_map


def fetch_forecast() -> tuple[pd.DataFrame, dict]:
    """
    Fetch hourly forecast + daily rain from Open-Meteo forecast API.
    Trimmed to FORECAST_END (midnight in 3 days).
    """
    params = {
        "latitude":      LAT,
        "longitude":     LON,
        "hourly":        "temperature_2m,shortwave_radiation",
        "daily":         "rain_sum",
        "timezone":      TIMEZONE,
        "forecast_days": 5,
    }
    print("  [2/2] Calling Open-Meteo forecast API (next 72h) ...")
    df, rain_map = _parse_response(requests.get(FORECAST_URL, params=params, timeout=60))

    # Trim hourly to exactly midnight 3 days from now
    df = df[df["DateTime"] <= FORECAST_END].reset_index(drop=True)
    print(f"        {len(df):,} hourly records  "
          f"({df['DateTime'].iloc[0]}  to  {df['DateTime'].iloc[-1]})")
    print(f"        {len(rain_map)} daily rain_sum values")
    return df, rain_map


# =============================================================================
# INTERPOLATION: hourly -> 5-minute
# =============================================================================

def interpolate_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate hourly DataFrame to 5-minute resolution using cubic splines.

    Parameters
    ----------
    df : DataFrame with columns [DateTime, Temperature_C, Irradiance_Wm2]

    Returns
    -------
    DataFrame at 5-minute resolution.
    """
    df     = df.sort_values("DateTime").reset_index(drop=True)
    origin = df["DateTime"].iloc[0]

    t_hours = np.array([(dt - origin).total_seconds() / 60.0
                         for dt in df["DateTime"]])
    t_5min  = np.arange(t_hours[0], t_hours[-1] + 1, 5.0)

    out = {"DateTime": [origin + datetime.timedelta(minutes=float(m))
                        for m in t_5min]}

    for col in ["Temperature_C", "Irradiance_Wm2"]:
        f        = interp1d(t_hours, df[col].values.astype(float),
                            kind="cubic", fill_value="extrapolate")
        interped = f(t_5min)
        if col == "Irradiance_Wm2":
            interped = np.maximum(interped, 0.0)
        out[col] = np.round(interped, 2)

    return pd.DataFrame(out)


# =============================================================================
# FORMAT OUTPUT — match Data_Prediction.xlsx style
# =============================================================================

def add_rain_columns(df: pd.DataFrame,
                     rain_hist: dict,
                     rain_fc: dict) -> pd.DataFrame:
    """
    Broadcast daily rain sum to every 5-minute row in a single column.

    Historical (measured) values are used for past days.
    Forecast values fill in for future days where no measurement exists.

    Parameters
    ----------
    rain_hist : {date -> mm}  from the archive API (measured)
    rain_fc   : {date -> mm}  from the forecast API
    """
    df    = df.copy()
    dates = df["DateTime"].dt.date
    hist  = dates.map(rain_hist)   # NaN for future dates
    fc    = dates.map(rain_fc)     # NaN for past dates not in forecast
    # Historical first, forecast fills the gap for future days
    df["Rain_Sum_mm"] = hist.fillna(fc).round(2)
    return df


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format columns to match Data_Prediction.xlsx style:
      Time      : DD.MM.YYYY HH:MM:SS
      Date      : DD.MM.YYYY
      Time_Only : HH:MM:SS
    """
    df = df.copy()
    df["Time"]      = df["DateTime"].dt.strftime("%d.%m.%Y %H:%M:%S")
    df["Date"]      = df["DateTime"].dt.strftime("%d.%m.%Y")
    df["Time_Only"] = df["DateTime"].dt.strftime("%H:%M:%S")
    return df[["Time", "Date", "Time_Only",
               "Temperature_C", "Irradiance_Wm2",
               "Rain_Sum_mm"]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Open-Meteo Fetch — Meiringen")
    print("=" * 60)
    print(f"  Historical : {START_DATE}  to  {TODAY}")
    print(f"  Forecast   : today  to  {FORECAST_END.strftime('%Y-%m-%d %H:%M')}  (+72h)")
    print(f"  Location   : lat={LAT}, lon={LON}  ({TIMEZONE})")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 60)

    # 1. Fetch historical + forecast (each returns hourly df + daily rain map)
    df_hist,     rain_hist = fetch_historical()
    df_forecast, rain_fc   = fetch_forecast()

    # 2. Combine hourly — historical takes priority for overlapping timestamps
    df_combined = (
        pd.concat([df_hist, df_forecast])
          .drop_duplicates(subset="DateTime", keep="first")
          .sort_values("DateTime")
          .reset_index(drop=True)
    )
    print(f"\n  Combined   : {len(df_combined):,} hourly records total")
    print(f"  Full range : {df_combined['DateTime'].iloc[0]}  to  "
          f"{df_combined['DateTime'].iloc[-1]}")

    # 3. Interpolate temperature + irradiance to 5-minute
    print("\n  Interpolating hourly -> 5-minute resolution ...")
    df_5min = interpolate_to_5min(df_combined)
    print(f"  {len(df_5min):,} rows at 5-min resolution")

    # 4. Broadcast daily rain sums to every 5-min row
    df_5min = add_rain_columns(df_5min, rain_hist, rain_fc)

    # 5. Format to match Data_Prediction.xlsx style
    df_out = format_output(df_5min)

    # 5. Save to Excel
    print(f"\n  Saving to Excel ...")
    df_out.to_excel(OUTPUT_FILE, index=False, sheet_name="Weather_Data")

    print()
    print("=" * 60)
    print(f"  Done!  {len(df_out):,} rows written.")
    print(f"  From : {df_out['Time'].iloc[0]}")
    print(f"  To   : {df_out['Time'].iloc[-1]}")
    print(f"  Cols : {list(df_out.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
