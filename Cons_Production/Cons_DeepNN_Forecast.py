"""
Real Forecasting Script for Consumption Prediction
===================================================
This script loads a pre-trained DeepNN model and makes predictions for a future date
where actual consumption values are NOT available.

WORKFLOW:
1. Run Cons_DeepNN_Prediction.py with SAVE_MODEL=True to train and save the model
2. Run this script to forecast consumption for a future date

The model file (Cons_DeepNN_Model.npz) contains:
- Neural network weights and biases
- Scaler parameters for feature and target normalization
- Model configuration (architecture, lookback steps)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from dnn_app_utils_v3 import *

# ========================================================================
# FORECAST CONFIGURATION
# ========================================================================
FORECAST_DATE = '2026-01-28'          # Date to forecast (no actual values needed)
MODEL_FILE = 'Cons_DeepNN_Model.npz'  # Saved model file from training
DATA_FILE = 'Data_January.csv'        # Data file with historical data + temperature forecasts
# ========================================================================

print("="*70)
print("REAL CONSUMPTION FORECASTING")
print(f"Forecast date: {FORECAST_DATE}")
print("="*70)

# 1 - LOAD THE TRAINED MODEL
print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found!\n"
                           f"Please run Cons_DeepNN_Prediction.py with SAVE_MODEL=True first.")

# Load model data
model_data = np.load(MODEL_FILE, allow_pickle=True)

# Extract model architecture
LAYERS_DIMS = model_data['layers_dims'].tolist()
LOOKBACK_STEPS = int(model_data['lookback_steps'])

print(f"Model architecture: {LAYERS_DIMS}")
print(f"Lookback steps: {LOOKBACK_STEPS} ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")

# Reconstruct parameters dictionary
parameters = {}
for l in range(1, len(LAYERS_DIMS)):
    parameters[f'W{l}'] = model_data[f'W{l}']
    parameters[f'b{l}'] = model_data[f'b{l}']

print(f"Loaded {len(parameters) // 2} layers of weights and biases")

# Reconstruct scalers
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.data_min_ = model_data['scaler_X_min']
scaler_X.data_max_ = model_data['scaler_X_max']
scaler_X.scale_ = model_data['scaler_X_scale']
scaler_X.data_range_ = model_data['scaler_X_data_range']
scaler_X.n_features_in_ = len(model_data['scaler_X_min'])
scaler_X.feature_range = (0, 1)
scaler_X.min_ = 0 - model_data['scaler_X_min'] * scaler_X.scale_

scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.data_min_ = model_data['scaler_y_min']
scaler_y.data_max_ = model_data['scaler_y_max']
scaler_y.scale_ = model_data['scaler_y_scale']
scaler_y.data_range_ = model_data['scaler_y_data_range']
scaler_y.n_features_in_ = 1
scaler_y.feature_range = (0, 1)
scaler_y.min_ = 0 - model_data['scaler_y_min'] * scaler_y.scale_

# Print training performance
print(f"\nTraining performance (from saved model):")
print(f"  R²:   {float(model_data['training_r2']):.4f}")
print(f"  MAE:  {float(model_data['training_mae']):.2f} kW")
print(f"  RMSE: {float(model_data['training_rmse']):.2f} kW")

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

# Get temperature forecasts for the forecast date (if available in data)
forecast_date_data = data[data['Date'] == forecast_date].copy()
if len(forecast_date_data) > 0:
    temp_forecast_dict = dict(zip(forecast_date_data['DateTime'],
                                   forecast_date_data['Temperature_Predicted']))
    print(f"Temperature forecasts found for {FORECAST_DATE}: {len(temp_forecast_dict)} values")
else:
    # Use last known temperature if no forecasts available
    last_temp = historical_data['Temperature_Predicted'].iloc[-1]
    temp_forecast_dict = {}
    print(f"No temperature forecasts found, will use last known value: {last_temp:.1f}°C")

# 4 - CREATE FORECAST FEATURES
print("\n" + "="*70)
print("CREATING FORECAST FEATURES")
print("="*70)

# Generate timestamps for the full forecast day (00:00 to 23:55)
forecast_start = pd.Timestamp(f"{FORECAST_DATE} 00:00:00")
forecast_end = pd.Timestamp(f"{FORECAST_DATE} 23:55:00")
forecast_timestamps = pd.date_range(start=forecast_start, end=forecast_end, freq='5min')

print(f"Generating forecasts for {len(forecast_timestamps)} time intervals")

# Create features for each forecast timestamp
X_forecast_list = []
valid_timestamps = []

for pred_ts in forecast_timestamps:
    # Calculate the required lookback window
    lookback_end_time = pred_ts
    lookback_start_time = pred_ts - pd.Timedelta(minutes=5 * LOOKBACK_STEPS)

    # Get the lookback window from historical data
    lookback_window = historical_data[
        (historical_data['DateTime'] > lookback_start_time) &
        (historical_data['DateTime'] < lookback_end_time)
    ].tail(LOOKBACK_STEPS)

    if len(lookback_window) < LOOKBACK_STEPS:
        print(f"  Skipping {pred_ts.strftime('%H:%M')} - insufficient lookback data ({len(lookback_window)}/{LOOKBACK_STEPS})")
        continue

    feature_row = []

    # Past consumption (LOOKBACK_STEPS values)
    past_consumption = lookback_window['Consumption'].values
    feature_row.extend(past_consumption)

    # Past temperature (LOOKBACK_STEPS values)
    past_temperature = lookback_window['Temperature'].values
    feature_row.extend(past_temperature)

    # Past predicted temperature (LOOKBACK_STEPS values)
    past_temp_pred = lookback_window['Temperature_Predicted'].values
    feature_row.extend(past_temp_pred)

    # Future predicted temperature for this timestamp
    if pred_ts in temp_forecast_dict:
        future_temp_pred = temp_forecast_dict[pred_ts]
    else:
        # Use last known temperature forecast
        future_temp_pred = historical_data['Temperature_Predicted'].iloc[-1]
    feature_row.append(future_temp_pred)

    # Time features
    feature_row.append(pred_ts.hour)
    feature_row.append(pred_ts.minute)
    feature_row.append(pred_ts.dayofweek)
    feature_row.append(pred_ts.month)
    feature_row.append(1 if pred_ts.dayofweek >= 5 else 0)

    X_forecast_list.append(feature_row)
    valid_timestamps.append(pred_ts)

X_forecast = np.array(X_forecast_list)
print(f"Created features for {len(valid_timestamps)} intervals")

# 5 - MAKE PREDICTIONS
print("\n" + "="*70)
print("GENERATING FORECASTS")
print("="*70)

# Normalize features using the saved scaler
X_forecast_scaled = scaler_X.transform(X_forecast)

# Reshape for neural network (features, samples)
X_forecast_nn = X_forecast_scaled.T

# Make predictions
dummy_y = np.zeros((1, X_forecast_nn.shape[1]))  # Not used for prediction
predictions_scaled = predict(X_forecast_nn, dummy_y, parameters)

# Inverse transform to get actual values
predictions = scaler_y.inverse_transform(predictions_scaled.flatten().reshape(-1, 1)).flatten()

print(f"Generated {len(predictions)} consumption forecasts")

# 6 - CREATE RESULTS
print("\n" + "="*70)
print(f"CONSUMPTION FORECAST FOR {FORECAST_DATE}")
print("="*70)

results_df = pd.DataFrame({
    'DateTime': valid_timestamps,
    'Time': [ts.strftime('%H:%M') for ts in valid_timestamps],
    'Predicted_Consumption_kW': predictions
})

print(results_df.to_string(index=False))

# Save to CSV
results_filename = f'Cons_DeepNN_Forecast_{FORECAST_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nForecast saved to: {results_filename}")

# 7 - PLOT FORECAST
print("\n" + "="*70)
print("GENERATING FORECAST PLOT")
print("="*70)

fig, ax = plt.subplots(1, 1, figsize=(16, 6))

x_vals = np.arange(len(valid_timestamps))

ax.plot(x_vals, predictions, 'b-', label='Predicted Consumption', linewidth=2)
ax.fill_between(x_vals, 0, predictions, alpha=0.3, color='blue')

ax.set_xlabel('Time of Day', fontsize=12)
ax.set_ylabel('Consumption (kW)', fontsize=12)
ax.set_title(f'Consumption FORECAST for {FORECAST_DATE} (5-min resolution)\n'
             f'Model Training R²: {float(model_data["training_r2"]):.4f}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# X-axis labels every hour
tick_positions = np.arange(0, len(valid_timestamps), 12)
tick_labels = [valid_timestamps[i].strftime('%H:%M') for i in tick_positions if i < len(valid_timestamps)]
ax.set_xticks(tick_positions[:len(tick_labels)])
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

plt.tight_layout()
plot_filename = f'Cons_DeepNN_Forecast_{FORECAST_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Forecast plot saved to: {plot_filename}")
plt.show()

# 8 - SUMMARY
print("\n" + "#"*70)
print("### FORECAST SUMMARY")
print("#"*70)

print(f"\nForecast Date: {FORECAST_DATE}")
print(f"Intervals Forecasted: {len(predictions)}")

print(f"\n--- Daily Forecast ---")
print(f"Total Predicted Consumption: {predictions.sum() / 12:.2f} kWh")
print(f"Average Predicted Power:     {predictions.mean():.2f} kW")
print(f"Peak Predicted Power:        {predictions.max():.2f} kW at {valid_timestamps[predictions.argmax()].strftime('%H:%M')}")
print(f"Min Predicted Power:         {predictions.min():.2f} kW at {valid_timestamps[predictions.argmin()].strftime('%H:%M')}")

print(f"\n--- Model Info ---")
print(f"Model file: {MODEL_FILE}")
print(f"Training R²: {float(model_data['training_r2']):.4f}")

print("\n" + "#"*70)
print("### FORECAST COMPLETE!")
print("#"*70)
