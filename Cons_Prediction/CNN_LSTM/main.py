"""
main.py
=======
Full daily pipeline — no email, no Task Scheduler dependency.
Run this manually (or via any scheduler) once per day.

Workflow:
  Step 1 — get_load_data      : extend calendar rows + import SCADA CSV files
  Step 2 — get_weather_data   : fetch latest weather from Open-Meteo API
  Step 3 — CNN_LSTM_Prediction: run 48h forecast (96h on Fridays)

Output CSV : Output Forecast/Prediction_<date>.csv
Log file   : main_runner.log
"""

import os
import sys
import datetime
import logging
import traceback


# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICT_DATE_OFFSET    = 0     # 0 = today at 00:00,  1 = tomorrow, etc.
RUN_ON_WEEKEND         = True  # If False, skip Saturday and Sunday
DEFAULT_FORECAST_HOURS = 48    # Mon–Thu
FRIDAY_FORECAST_HOURS  = 96   # Friday (covers full weekend)

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Output Forecast")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
LOG_FILE   = os.path.join(SCRIPT_DIR, "main_runner.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# PIPELINE
# =============================================================================

def _verify_output(predict_date: str) -> str:
    export_date  = (datetime.date.fromisoformat(predict_date) + datetime.timedelta(days=1)).isoformat()
    path = os.path.join(OUTPUT_DIR, f"Prediction_{export_date}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected forecast CSV not found: {path}")
    return path


def main():
    today = datetime.date.today()

    if not RUN_ON_WEEKEND and today.weekday() >= 5:
        log.info(f"Skipping — today is {today.strftime('%A')} and RUN_ON_WEEKEND=False.")
        return

    predict_date   = (today + datetime.timedelta(days=PREDICT_DATE_OFFSET)).strftime("%Y-%m-%d")
    forecast_hours = FRIDAY_FORECAST_HOURS if today.weekday() == 4 else DEFAULT_FORECAST_HOURS

    log.info("=" * 60)
    log.info("  AlpenEnergie — Main Forecast Pipeline")
    log.info(f"  Predict date   : {predict_date}")
    log.info(f"  Forecast hours : {forecast_hours}h")
    log.info(f"  Started        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    start = datetime.datetime.now()

    # Step 1 — Load data (extend calendar + import SCADA CSVs)
    log.info("Step 1/3 — Load data ...")
    try:
        import get_load_data
        get_load_data.main()
        log.info("  Step 1 — done.")
    except Exception:
        log.error(f"Step 1 failed:\n{traceback.format_exc()}")
        sys.exit(1)

    # Step 2 — Weather update
    log.info("Step 2/3 — Weather update ...")
    try:
        import get_weather_data
        get_weather_data.main()
        log.info("  Step 2 — done.")
    except Exception:
        log.error(f"Step 2 failed:\n{traceback.format_exc()}")
        sys.exit(1)

    # Step 3 — Forecast
    log.info(f"Step 3/3 — CNN-LSTM {forecast_hours}h forecast ...")
    try:
        import CNN_LSTM_Prediction
        CNN_LSTM_Prediction.run_forecast(predict_date, hours=forecast_hours)
        output_path = _verify_output(predict_date)
        log.info(f"  Output: {output_path}")
        log.info("  Step 3 — done.")
    except Exception:
        log.error(f"Step 3 failed:\n{traceback.format_exc()}")
        sys.exit(1)

    duration = (datetime.datetime.now() - start).total_seconds()
    log.info("=" * 60)
    log.info(f"  Pipeline complete in {duration:.0f}s")
    log.info(f"  Result: {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
