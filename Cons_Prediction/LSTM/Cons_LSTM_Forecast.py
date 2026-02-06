"""
Real Forecasting Script for LSTM Consumption Prediction
=======================================================
This script loads a pre-trained LSTM model and makes predictions for a future date
where actual consumption values are NOT available.

WORKFLOW:
1. Run Cons_LSTM_Prediction.py with SAVE_MODEL=True to train and save the model
2. Run this script to forecast consumption for a future date

The saved files contain:
- Cons_LSTM_Model.keras: The trained Keras LSTM model
- Cons_LSTM_Config.npz: Scaler parameters and model configuration
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ========================================================================
# FORECAST CONFIGURATION
# ========================================================================
FORECAST_DATE = '2026-01-28'              # Date to forecast (no actual values needed)
MODEL_FILE = 'Cons_LSTM_Model.keras'      # Saved Keras model (in same folder)
CONFIG_FILE = 'Cons_LSTM_Config.npz'      # Saved configuration and scalers
DATA_FILE = '../Data_January.csv'         # Data file (in parent folder)
# ========================================================================

print("="*70)
print("REAL LSTM CONSUMPTION FORECASTING")
print(f"Forecast date: {FORECAST_DATE}")
print("="*70)

# 1 - LOAD THE TRAINED MODEL AND CONFIGURATION
print("\n" + "="*70)
print("LOADING TRAINED LSTM MODEL")
print("="*70)

# Check files exist
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found!\n"
                           f"Please run Cons_LSTM_Prediction.py with SAVE_MODEL=True first.")
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Config file '{CONFIG_FILE}' not found!\n"
                           f"Please run Cons_LSTM_Prediction.py with SAVE_MODEL=True first.")

# Load Keras model
model = load_model(MODEL_FILE)
print(f"Loaded Keras model from: {MODEL_FILE}")

# Load configuration
config = np.load(CONFIG_FILE, allow_pickle=True)

LOOKBACK_STEPS = int(config['lookback_steps'])
LSTM_UNITS = int(config['lstm_units'])
FEATURE_COLUMNS = config['feature_columns'].tolist()
n_features = int(config['n_features'])

# Check if model uses future temperature forecast
if 'use_future_temp' in config:
    USE_FUTURE_TEMP = bool(config['use_future_temp'])
else:
    # Fallback: check if Future_Temp_Forecast is in feature columns
    USE_FUTURE_TEMP = 'Future_Temp_Forecast' in FEATURE_COLUMNS

if 'sequence_features' in config:
    SEQUENCE_FEATURES = config['sequence_features'].tolist()
elif USE_FUTURE_TEMP:
    # Sequence features are all except the future temp
    SEQUENCE_FEATURES = [f for f in FEATURE_COLUMNS if f != 'Future_Temp_Forecast']
else:
    SEQUENCE_FEATURES = FEATURE_COLUMNS

print(f"Lookback steps: {LOOKBACK_STEPS} ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
print(f"LSTM Units: {LSTM_UNITS}")
print(f"Features: {FEATURE_COLUMNS}")
print(f"Uses future temperature forecast: {USE_FUTURE_TEMP}")

# Reconstruct scalers
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.data_min_ = config['scaler_X_min']
scaler_X.data_max_ = config['scaler_X_max']
scaler_X.scale_ = config['scaler_X_scale']
scaler_X.data_range_ = config['scaler_X_data_range']
scaler_X.n_features_in_ = len(config['scaler_X_min'])
scaler_X.feature_range = (0, 1)
scaler_X.min_ = 0 - config['scaler_X_min'] * scaler_X.scale_

scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.data_min_ = config['scaler_y_min']
scaler_y.data_max_ = config['scaler_y_max']
scaler_y.scale_ = config['scaler_y_scale']
scaler_y.data_range_ = config['scaler_y_data_range']
scaler_y.n_features_in_ = 1
scaler_y.feature_range = (0, 1)
scaler_y.min_ = 0 - config['scaler_y_min'] * scaler_y.scale_

# Print training performance
print(f"\nTraining performance (from saved model):")
print(f"  R²:   {float(config['training_r2']):.4f}")
print(f"  MAE:  {float(config['training_mae']):.2f} kW")
print(f"  RMSE: {float(config['training_rmse']):.2f} kW")
print(f"  Epochs trained: {int(config['epochs_trained'])}")

# 2 - LOAD AND PREPROCESS DATA
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load data from CSV
data = pd.read_csv(DATA_FILE, skiprows=3, header=None, encoding='latin-1')

# Assign column names
data.columns = ['DateTime_str', 'Date', 'DayTime', 'Forecast_Prod', 'Forecast_Load',
                'Consumption', 'Production', 'Level_Bidmi', 'Level_Haselholz',
                'Temperature', 'Irradiance', 'Rain', 'SDR_Mode', 'Forecast_Mode',
                'Transfer_Mode', 'Waterlevel_Mode', 'Temp_Forecast']

# Parse DateTime
data['DateTime'] = pd.to_datetime(data['DateTime_str'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
data = data.dropna(subset=['DateTime'])
data = data.sort_values('DateTime').reset_index(drop=True)

# Handle missing Temp_Forecast values
data['Temp_Forecast'] = pd.to_numeric(data['Temp_Forecast'], errors='coerce')
data['Temp_Forecast'] = data['Temp_Forecast'].fillna(data['Temperature'])
data.rename(columns={'Temp_Forecast': 'Temperature_Predicted'}, inplace=True)

# Add date column
data['Date'] = data['DateTime'].dt.date

# Add cyclical time features (same as training)
data['Hour'] = data['DateTime'].dt.hour
data['DayOfWeek'] = data['DateTime'].dt.dayofweek
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

print(f"Data loaded: {len(data)} rows")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

# 3 - PREPARE LOOKBACK DATA
print("\n" + "="*70)
print(f"PREPARING FORECAST FOR {FORECAST_DATE}")
print("="*70)

forecast_date = pd.to_datetime(FORECAST_DATE).date()

# Get data BEFORE the forecast date for lookback features
historical_data = data[data['Date'] < forecast_date].copy()
print(f"Historical data available: {len(historical_data)} rows")
if len(historical_data) > 0:
    print(f"Historical data range: {historical_data['DateTime'].min()} to {historical_data['DateTime'].max()}")
    print(f"Lookback required: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
else:
    raise ValueError("No historical data found before the forecast date!")

# Get temperature forecasts for the forecast date (if available in data)
forecast_date_data = data[data['Date'] == forecast_date].copy()
if len(forecast_date_data) > 0:
    print(f"Temperature forecasts found for {FORECAST_DATE}: {len(forecast_date_data)} values")
else:
    print(f"No temperature forecasts found for {FORECAST_DATE}")

# 4 - AUTOREGRESSIVE FORECASTING
print("\n" + "="*70)
print("AUTOREGRESSIVE FORECAST SEQUENCES")
print("="*70)

# Generate timestamps for the full forecast day (00:00 to 23:55)
forecast_start = pd.Timestamp(f"{FORECAST_DATE} 00:00:00")
forecast_end = pd.Timestamp(f"{FORECAST_DATE} 23:55:00")
forecast_timestamps = pd.date_range(start=forecast_start, end=forecast_end, freq='5min')

print(f"Generating forecasts for {len(forecast_timestamps)} time intervals")
print(f"Using AUTOREGRESSIVE approach: each prediction feeds into the next")

# Build a rolling buffer that we'll update with predictions
# Start with the last LOOKBACK_STEPS rows of historical data
rolling_buffer = historical_data.tail(LOOKBACK_STEPS).copy().reset_index(drop=True)

# Ensure we have enough data
if len(rolling_buffer) < LOOKBACK_STEPS:
    raise ValueError(f"Insufficient historical data: {len(rolling_buffer)}/{LOOKBACK_STEPS}")

predictions = []
valid_timestamps = []

print(f"\nStarting autoregressive forecasting...")

for i, pred_ts in enumerate(forecast_timestamps):
    # Get temperature forecast for this timestamp
    if len(forecast_date_data) > 0:
        temp_row = forecast_date_data[forecast_date_data['DateTime'] == pred_ts]
        if len(temp_row) > 0:
            future_temp = temp_row['Temperature_Predicted'].values[0]
            # Check if actual temperature is available (not NaN)
            temp_val = temp_row['Temperature'].values[0]
            temp_actual = temp_val if pd.notna(temp_val) else future_temp
        else:
            future_temp = rolling_buffer['Temperature_Predicted'].iloc[-1]
            temp_actual = rolling_buffer['Temperature'].iloc[-1]
    else:
        future_temp = rolling_buffer['Temperature_Predicted'].iloc[-1]
        temp_actual = rolling_buffer['Temperature'].iloc[-1]

    # Build the sequence from the rolling buffer
    sequence_data = rolling_buffer[SEQUENCE_FEATURES].values.copy()

    # Add future temperature column if model uses it
    if USE_FUTURE_TEMP:
        future_temp_col = np.full((LOOKBACK_STEPS, 1), future_temp)
        sequence_data = np.hstack([sequence_data, future_temp_col])

    # Reshape for scaler: (timesteps, features) -> need to process as 2D
    sequence_reshaped = sequence_data.reshape(-1, n_features)

    # Normalize
    sequence_scaled = scaler_X.transform(sequence_reshaped)

    # Reshape for LSTM: (1, timesteps, features)
    sequence_scaled = sequence_scaled.reshape(1, LOOKBACK_STEPS, n_features)

    # Make prediction
    pred_scaled = model.predict(sequence_scaled, verbose=0)

    # Inverse transform to get actual consumption value
    pred_consumption = scaler_y.inverse_transform(pred_scaled).flatten()[0]

    predictions.append(pred_consumption)
    valid_timestamps.append(pred_ts)

    # Now update the rolling buffer for the next prediction
    # Create a new row with the predicted consumption and proper time features
    new_row = pd.DataFrame({
        'DateTime': [pred_ts],
        'Consumption': [pred_consumption],
        'Temperature': [temp_actual],
        'Temperature_Predicted': [future_temp],
        'Hour_sin': [np.sin(2 * np.pi * pred_ts.hour / 24)],
        'Hour_cos': [np.cos(2 * np.pi * pred_ts.hour / 24)],
        'DayOfWeek_sin': [np.sin(2 * np.pi * pred_ts.dayofweek / 7)],
        'DayOfWeek_cos': [np.cos(2 * np.pi * pred_ts.dayofweek / 7)],
        'IsWeekend': [1 if pred_ts.dayofweek >= 5 else 0]
    })

    # Remove the oldest row and append the new prediction
    rolling_buffer = pd.concat([rolling_buffer.iloc[1:], new_row], ignore_index=True)

    # Progress update every hour
    if (i + 1) % 12 == 0:
        print(f"  {pred_ts.strftime('%H:%M')} - Predicted: {pred_consumption:.1f} kW")

predictions = np.array(predictions)
print(f"\nGenerated {len(predictions)} consumption forecasts")

print(f"Generated {len(predictions)} consumption forecasts")

# 6 - CREATE RESULTS
print("\n" + "="*70)
print(f"LSTM CONSUMPTION FORECAST FOR {FORECAST_DATE}")
print("="*70)

results_df = pd.DataFrame({
    'DateTime': valid_timestamps,
    'Time': [ts.strftime('%H:%M') for ts in valid_timestamps],
    'Predicted_Consumption_kW': predictions
})

print(results_df.to_string(index=False))

# Save to CSV
results_filename = f'Cons_LSTM_Forecast_{FORECAST_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nForecast saved to: {results_filename}")

# 7 - PLOT FORECAST WITH PREVIOUS DAY COMPARISON
print("\n" + "="*70)
print("GENERATING FORECAST PLOT WITH PREVIOUS DAY COMPARISON")
print("="*70)

# Get previous day's actual consumption for comparison
prev_date = forecast_date - pd.Timedelta(days=1)
prev_day_data = historical_data[historical_data['Date'] == prev_date].copy()

if len(prev_day_data) > 0:
    prev_consumption = prev_day_data['Consumption'].values
    prev_times = prev_day_data['DateTime'].dt.strftime('%H:%M').values
    print(f"Previous day ({prev_date}) data found: {len(prev_day_data)} values")
else:
    prev_consumption = None
    print(f"No previous day data found for comparison")

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

x_vals = np.arange(len(valid_timestamps))

# Plot forecast
ax.plot(x_vals, predictions, 'b-', label=f'Forecast {FORECAST_DATE}', linewidth=2.5)
ax.fill_between(x_vals, predictions.min() * 0.95, predictions, alpha=0.2, color='blue')

# Plot previous day actual consumption if available
if prev_consumption is not None and len(prev_consumption) == len(predictions):
    ax.plot(x_vals, prev_consumption, 'g--', label=f'Actual {prev_date}', linewidth=2, alpha=0.8)
    ax.fill_between(x_vals, predictions.min() * 0.95, prev_consumption, alpha=0.1, color='green')

# Set Y-axis scale to show variation better (not starting from 0)
all_values = list(predictions)
if prev_consumption is not None and len(prev_consumption) == len(predictions):
    all_values.extend(prev_consumption)
y_min = min(all_values) * 0.9
y_max = max(all_values) * 1.05
ax.set_ylim(y_min, y_max)

ax.set_xlabel('Time of Day', fontsize=12)
ax.set_ylabel('Consumption (kW)', fontsize=12)
ax.set_title(f'LSTM Consumption FORECAST for {FORECAST_DATE} vs Previous Day\n'
             f'Model Training R²: {float(config["training_r2"]):.4f}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

# X-axis labels every 2 hours
tick_positions = np.arange(0, len(valid_timestamps), 24)  # Every 2 hours (24 x 5min = 2h)
tick_labels = [valid_timestamps[i].strftime('%H:%M') for i in tick_positions if i < len(valid_timestamps)]
ax.set_xticks(tick_positions[:len(tick_labels)])
ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)

# Add minor gridlines
ax.minorticks_on()
ax.grid(which='minor', alpha=0.15)

plt.tight_layout()
plot_filename = f'Cons_LSTM_Forecast_{FORECAST_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Forecast plot saved to: {plot_filename}")
plt.show()

# 8 - SUMMARY
print("\n" + "#"*70)
print("### LSTM FORECAST SUMMARY")
print("#"*70)

print(f"\nForecast Date: {FORECAST_DATE}")
print(f"Intervals Forecasted: {len(predictions)}")

print(f"\n--- Daily Forecast ({FORECAST_DATE}) ---")
print(f"Total Predicted Consumption: {predictions.sum() / 12:.2f} kWh")
print(f"Average Predicted Power:     {predictions.mean():.2f} kW")
print(f"Peak Predicted Power:        {predictions.max():.2f} kW at {valid_timestamps[predictions.argmax()].strftime('%H:%M')}")
print(f"Min Predicted Power:         {predictions.min():.2f} kW at {valid_timestamps[predictions.argmin()].strftime('%H:%M')}")

# Compare with previous day if available
if prev_consumption is not None and len(prev_consumption) == len(predictions):
    print(f"\n--- Previous Day ({prev_date}) Actual ---")
    print(f"Total Actual Consumption:    {prev_consumption.sum() / 12:.2f} kWh")
    print(f"Average Actual Power:        {prev_consumption.mean():.2f} kW")
    print(f"Peak Actual Power:           {prev_consumption.max():.2f} kW")
    print(f"Min Actual Power:            {prev_consumption.min():.2f} kW")

    print(f"\n--- Comparison (Forecast vs Previous Day) ---")
    total_diff = (predictions.sum() - prev_consumption.sum()) / 12
    total_diff_pct = (predictions.sum() - prev_consumption.sum()) / prev_consumption.sum() * 100
    print(f"Total Difference:            {total_diff:+.2f} kWh ({total_diff_pct:+.2f}%)")
    print(f"Average Difference:          {predictions.mean() - prev_consumption.mean():+.2f} kW")

    # Calculate correlation
    correlation = np.corrcoef(predictions, prev_consumption)[0, 1]
    print(f"Pattern Correlation:         {correlation:.4f}")

print(f"\n--- Model Info ---")
print(f"Model file: {MODEL_FILE}")
print(f"LSTM Units: {LSTM_UNITS}")
print(f"Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
print(f"Training R²: {float(config['training_r2']):.4f}")

print("\n" + "#"*70)
print("### LSTM FORECAST COMPLETE!")
print("#"*70)
