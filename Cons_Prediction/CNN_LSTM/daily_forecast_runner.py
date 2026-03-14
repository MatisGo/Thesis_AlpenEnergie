"""
daily_forecast_runner.py
========================
Automated daily forecast pipeline for CNN-LSTM load prediction.

Workflow:
  1. Fetch latest weather data   (get_weather_data.py)
  2. Run 48h load forecast       (CNN_LSTM_Prediction.py --predict <today 00:00>)
  3. Email the result CSV to configured recipients

Schedule this script via Windows Task Scheduler to run at 07:30 every morning.
On any failure, a notification email is sent automatically.

Gmail setup (one-time):
  1. Enable 2-Step Verification on your Google account
  2. Go to https://myaccount.google.com/apppasswords
  3. Create an App Password (select "Mail" + "Windows Computer")
  4. Paste the generated 16-character password into SENDER_APP_PASSWORD below
"""

import os
import sys
import datetime
import subprocess
import smtplib
import logging
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders


# =============================================================================
# CONFIGURATION — edit these values before first use
# =============================================================================

# --- Email -------------------------------------------------------------------
SENDER_EMAIL        = "matisgo.74@gmail.com"       # Gmail used to send
SENDER_APP_PASSWORD = "hkjk hszi rcqr vmge"         # Gmail App Password (16 chars)
RECIPIENT_EMAILS    = ["matis.gourdes@proton.me"] #, "timothee.devarax@gmail.com"]     # List — add more if needed

# --- Prediction --------------------------------------------------------------
PREDICT_DATE_OFFSET    = 0     # 0 = today at 00:00,  1 = tomorrow, etc.
RUN_ON_WEEKEND         = False # If False, script exits immediately on Saturday and Sunday
DEFAULT_FORECAST_HOURS = 48   # Mon–Thu forecast horizon
FRIDAY_FORECAST_HOURS  = 96   # Fri–Sun forecast horizon (covers the full weekend)

# --- SMTP (Gmail — no change needed) ----------------------------------------
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# --- Paths (auto-resolved relative to this script) ---------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
WEATHER_SCRIPT = os.path.join(SCRIPT_DIR, "get_weather_data.py")
PREDICT_SCRIPT = os.path.join(SCRIPT_DIR, "CNN_LSTM_Prediction.py")
RESULTS_DIR    = os.path.join(SCRIPT_DIR, "Simulation results")
LOG_FILE       = os.path.join(SCRIPT_DIR, "daily_runner.log")

# --- Python interpreter (same environment as this script) -------------------
PYTHON = sys.executable


# =============================================================================
# LOGGING — writes to both console and daily_runner.log
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
# EMAIL HELPERS
# =============================================================================

def _smtp_connect():
    """Open an authenticated SMTP connection to Gmail."""
    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
    server.ehlo()
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
    return server


def send_success_email(csv_path: str, predict_date: str, duration_s: float,
                       hours: int = DEFAULT_FORECAST_HOURS):
    """Send forecast CSV as attachment to all recipients."""
    subject = f"[AlpenEnergie / Master Thesis] {hours}h Load Forecast — {predict_date}"
    body = (
        f"Hello,\n\n"
        f"Here is the daily {hours}h load forecast for {predict_date} "
        f"completed in {duration_s:.0f} seconds.\n\n"
        f"The forecast CSV is attached.\n\n"
        f"Best Regards, \nMatis Gourdes\n\nAlpenEnergie Automated Forecast"
    )

    msg = MIMEMultipart()
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = ", ".join(RECIPIENT_EMAILS)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with open(csv_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{os.path.basename(csv_path)}"',
    )
    msg.attach(part)

    with _smtp_connect() as server:
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
    log.info(f"Success email sent to: {', '.join(RECIPIENT_EMAILS)}")


def send_failure_email(step: str, error_details: str):
    """Send failure notification email to all recipients."""
    subject = f"[AlpenEnergie] FORECAST FAILED — {datetime.date.today()}"
    body = (
        f"The daily forecast pipeline failed at step:\n"
        f"  {step}\n\n"
        f"Error details:\n"
        f"{error_details}\n\n"
        f"Log file: {LOG_FILE}\n\n"
        f"— AlpenEnergie Automated Forecast"
    )

    msg = MIMEMultipart()
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = ", ".join(RECIPIENT_EMAILS)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with _smtp_connect() as server:
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        log.info(f"Failure email sent to: {', '.join(RECIPIENT_EMAILS)}")
    except Exception as e:
        # Swallow — don't crash while reporting a crash
        log.error(f"Could not send failure email: {e}")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def run_weather_update():
    """Step 1 — Fetch latest weather data from Open-Meteo."""
    log.info("Step 1/2 — Updating weather data ...")
    result = subprocess.run(
        [PYTHON, WEATHER_SCRIPT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    log.info("  Weather update complete.")


def run_prediction(predict_date: str, hours: int) -> str:
    """
    Step 2 — Run CNN-LSTM forecast for predict_date (YYYY-MM-DD).
    Returns the path to the generated CSV file.
    """
    log.info(f"Step 2/2 — Running CNN-LSTM {hours}h forecast for {predict_date} ...")
    result = subprocess.run(
        [PYTHON, PREDICT_SCRIPT, "--predict", predict_date, "--hours", str(hours)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    log.info("  Forecast complete.")

    csv_filename = f"Prediction_{predict_date}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")
    return csv_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    today = datetime.date.today()

    # Weekend guard — Saturday=5, Sunday=6
    if not RUN_ON_WEEKEND and today.weekday() >= 5:
        log.info(f"  Skipping — today is {today.strftime('%A')} and RUN_ON_WEEKEND=False.")
        return

    predict_date = (
        today + datetime.timedelta(days=PREDICT_DATE_OFFSET)
    ).strftime("%Y-%m-%d")

    # Friday (weekday=4) triggers the 96h weekend forecast; all other days use 48h
    forecast_hours = FRIDAY_FORECAST_HOURS if today.weekday() == 4 else DEFAULT_FORECAST_HOURS

    log.info("=" * 60)
    log.info("  AlpenEnergie Daily Forecast Runner")
    log.info(f"  Predict date   : {predict_date}")
    log.info(f"  Forecast hours : {forecast_hours}h")
    log.info(f"  Started at     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    start_time = datetime.datetime.now()

    # Step 1 — Weather update
    try:
        run_weather_update()
    except Exception:
        details = traceback.format_exc()
        log.error(f"Weather update failed:\n{details}")
        send_failure_email("Step 1 — Weather Update (get_weather_data.py)", details)
        sys.exit(1)

    # Step 2 — Forecast
    try:
        csv_path = run_prediction(predict_date, forecast_hours)
    except Exception:
        details = traceback.format_exc()
        log.error(f"Forecast failed:\n{details}")
        send_failure_email("Step 2 — CNN-LSTM Prediction (CNN_LSTM_Prediction.py)", details)
        sys.exit(1)

    # Step 3 — Send result email
    duration = (datetime.datetime.now() - start_time).total_seconds()
    try:
        send_success_email(csv_path, predict_date, duration, forecast_hours)
    except Exception:
        details = traceback.format_exc()
        log.error(f"Email sending failed:\n{details}")
        # Forecast succeeded — do not exit with error code

    log.info("=" * 60)
    log.info(f"  Pipeline complete in {duration:.0f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
