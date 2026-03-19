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

    # Step 3 — Load forecast
    log.info(f"Step 3/4 — CNN-LSTM {forecast_hours}h forecast ...")
    try:
        import CNN_LSTM_Prediction
        CNN_LSTM_Prediction.run_forecast(predict_date, hours=forecast_hours)
        output_path = _verify_output(predict_date)
        log.info(f"  Output: {output_path}")
        log.info("  Step 3 — done.")
    except Exception:
        log.error(f"Step 3 failed:\n{traceback.format_exc()}")
        sys.exit(1)

    # Step 4 — Production forecast (DNN) → write to new sheet in output Excel
    log.info("Step 4/4 — DNN production forecast ...")
    try:
        import DNN_Prod_Prediction
        prod_value, prod_date = DNN_Prod_Prediction.run_predict(show_plots=False)

        if prod_value is not None:
            import pandas as pd

            # Convert output CSV to Excel if needed, then add Prod Prediction sheet
            xlsx_path = output_path.replace('.csv', '.xlsx')
            df_load = pd.read_csv(output_path)

            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                df_load.to_excel(writer, sheet_name='Load Forecast', index=False)
                df_prod = pd.DataFrame({
                    'Date':                   [str(prod_date.date())],
                    'Predicted_Avg_Prod_kW':  [round(float(prod_value), 1)],
                })
                df_prod.to_excel(writer, sheet_name='Prod Prediction', index=False)

            log.info(f"  Production forecast: {prod_value:,.1f} kW avg for {prod_date.date()}")
            log.info(f"  Output (Excel): {xlsx_path}")
        else:
            log.warning("  Production forecast skipped (no model or weather data).")
        log.info("  Step 4 — done.")
    except Exception:
        log.error(f"Step 4 failed:\n{traceback.format_exc()}")
        # Non-critical — do not exit

    duration = (datetime.datetime.now() - start).total_seconds()
    log.info("=" * 60)
    log.info(f"  Pipeline complete in {duration:.0f}s")
    log.info(f"  Result: {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
