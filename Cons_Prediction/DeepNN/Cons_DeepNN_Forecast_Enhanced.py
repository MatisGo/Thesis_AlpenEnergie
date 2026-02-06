"""
Enhanced DeepNN Forecasting Script for Consumption Prediction
=============================================================
This script loads a pre-trained DeepNN model and makes predictions for a future date
with VARIANCE PRESERVATION techniques to avoid flat/smooth predictions.

ENHANCEMENT MODES:
- 'none': Raw DeepNN predictions (smooth, may not match historical variance)
- 'residual_boost': Add random noise based on historical residuals
- 'pattern_match': Blend with similar historical days' patterns
- 'volatility_scale': Scale predictions to match historical volatility
- 'hybrid': Use historical pattern as BASE, DeepNN provides adjustment (RECOMMENDED)

WORKFLOW:
1. Run Cons_DeepNN_Prediction.py with SAVE_MODEL=True to train and save the model
2. Run this script to forecast consumption with variance preservation for a future date
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Import DeepNN utilities from parent directory
import sys
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dnn_app_utils_v3 import *

# ========================================================================
# FORECAST CONFIGURATION
# ========================================================================
FORECAST_DATE = '2026-01-28'              # Date to forecast
MODEL_FILE = 'Cons_DeepNN_Model.npz'      # Saved DeepNN model
DATA_FILE = '../Data_January.csv'         # Data file (relative to this folder)

# ENHANCEMENT OPTIONS
# Options: 'none', 'residual_boost', 'pattern_match', 'volatility_scale', 'hybrid'
VARIANCE_MODE = 'hybrid'         # 'hybrid' = use historical pattern as BASE, DeepNN as adjustment
BLEND_RATIO = 0.85               # For 'hybrid': weight of historical pattern (0.85 = 85% history, 15% DeepNN)
SIMILAR_DAYS_COUNT = 5           # Number of similar days to use for pattern matching (more = smoother average)
COMPARISON_DAYS = 4              # Number of past days to show in comparison plot
# ========================================================================

print("="*70)
print("ENHANCED DEEPNN CONSUMPTION FORECASTING")
print(f"Forecast date: {FORECAST_DATE}")
print(f"Variance preservation mode: {VARIANCE_MODE}")
print("="*70)

# 1 - LOAD THE TRAINED MODEL
print("\n" + "="*70)
print("LOADING TRAINED DEEPNN MODEL")
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

# 3 - PREPARE LOOKBACK DATA AND FIND SIMILAR DAYS
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
    print(f"Temperature forecasts found for {FORECAST_DATE}: {len(forecast_date_data)} values")
else:
    print(f"No temperature forecasts found for {FORECAST_DATE}")

# 4 - FIND SIMILAR HISTORICAL DAYS FOR PATTERN MATCHING
print("\n" + "="*70)
print(f"FINDING {SIMILAR_DAYS_COUNT} SIMILAR HISTORICAL DAYS")
print("="*70)

forecast_dt = pd.to_datetime(FORECAST_DATE)
forecast_dayofweek = forecast_dt.dayofweek
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
print(f"Forecast day: {day_names[forecast_dayofweek]}")

# Get all complete days with 288 intervals
historical_data['DateOnly'] = historical_data['DateTime'].dt.date
daily_counts = historical_data.groupby('DateOnly').size()
complete_days = daily_counts[daily_counts >= 288].index.tolist()

# Find similar days (same day of week)
similar_days = []
for d in complete_days:
    d_dt = pd.to_datetime(d)
    if d_dt.dayofweek == forecast_dayofweek:
        similar_days.append(d)

# Take the most recent similar days
similar_days = sorted(similar_days, reverse=True)[:SIMILAR_DAYS_COUNT]
print(f"Found {len(similar_days)} similar days ({day_names[forecast_dayofweek]}s):")
for d in similar_days:
    print(f"  - {d}")

# Extract patterns from similar days
similar_patterns = []
for d in similar_days:
    day_data = historical_data[historical_data['DateOnly'] == d].sort_values('DateTime')
    if len(day_data) >= 288:
        similar_patterns.append(day_data['Consumption'].values[:288])

if len(similar_patterns) > 0:
    avg_pattern = np.mean(similar_patterns, axis=0)
    std_pattern = np.std(similar_patterns, axis=0)
    print(f"\nSimilar days pattern statistics:")
    print(f"  Mean consumption: {avg_pattern.mean():.1f} kW")
    print(f"  Pattern std dev: {std_pattern.mean():.1f} kW")
else:
    avg_pattern = None
    std_pattern = None
    print("No similar day patterns found!")

# 5 - AUTOREGRESSIVE FORECASTING WITH VARIANCE ENHANCEMENT
print("\n" + "="*70)
print("AUTOREGRESSIVE FORECASTING WITH VARIANCE ENHANCEMENT")
print("="*70)

# Generate timestamps for the full forecast day
forecast_start = pd.Timestamp(f"{FORECAST_DATE} 00:00:00")
forecast_end = pd.Timestamp(f"{FORECAST_DATE} 23:55:00")
forecast_timestamps = pd.date_range(start=forecast_start, end=forecast_end, freq='5min')

print(f"Generating forecasts for {len(forecast_timestamps)} time intervals")

# Initialize rolling buffer with last LOOKBACK_STEPS rows of historical data
rolling_buffer = historical_data.tail(LOOKBACK_STEPS).copy()
rolling_buffer = rolling_buffer.reset_index(drop=True)

# Historical statistics for variance enhancement
hist_by_time = historical_data.groupby(historical_data['DateTime'].dt.time)['Consumption'].agg(['mean', 'std'])

predictions_raw = []  # Raw DeepNN predictions
predictions_enhanced = []  # Enhanced with variance
valid_timestamps = []

print(f"\nStarting autoregressive forecasting with {VARIANCE_MODE} enhancement...")
print(f"BLEND_RATIO = {BLEND_RATIO} ({'%.0f%% historical pattern, %.0f%% DeepNN' % (BLEND_RATIO*100, (1-BLEND_RATIO)*100) if VARIANCE_MODE == 'hybrid' else ''})")

# Debug: Show initial buffer statistics
print(f"\nInitial rolling buffer statistics:")
print(f"  Last consumption values: {rolling_buffer['Consumption'].tail(5).values}")
print(f"  Buffer consumption mean: {rolling_buffer['Consumption'].mean():.1f} kW")
if len(similar_patterns) > 0:
    print(f"  Similar days pattern mean: {avg_pattern.mean():.1f} kW")
    print(f"  Similar days pattern range: {avg_pattern.min():.1f} - {avg_pattern.max():.1f} kW")

for i, pred_ts in enumerate(forecast_timestamps):
    # Get temperature forecast for this timestamp
    if len(forecast_date_data) > 0:
        temp_row = forecast_date_data[forecast_date_data['DateTime'] == pred_ts]
        if len(temp_row) > 0:
            future_temp = temp_row['Temperature_Predicted'].values[0]
            temp_val = temp_row['Temperature'].values[0]
            temp_actual = temp_val if pd.notna(temp_val) else future_temp
        else:
            future_temp = historical_data['Temperature_Predicted'].iloc[-1]
            temp_actual = future_temp
    else:
        future_temp = historical_data['Temperature_Predicted'].iloc[-1]
        temp_actual = future_temp

    # Build feature vector from rolling buffer
    feature_row = []

    # Past consumption (LOOKBACK_STEPS values)
    past_consumption = rolling_buffer['Consumption'].values
    feature_row.extend(past_consumption)

    # Past temperature (LOOKBACK_STEPS values)
    past_temperature = rolling_buffer['Temperature'].values
    feature_row.extend(past_temperature)

    # Past predicted temperature (LOOKBACK_STEPS values)
    past_temp_pred = rolling_buffer['Temperature_Predicted'].values
    feature_row.extend(past_temp_pred)

    # Future predicted temperature
    feature_row.append(future_temp)

    # Time features
    feature_row.append(pred_ts.hour)
    feature_row.append(pred_ts.minute)
    feature_row.append(pred_ts.dayofweek)
    feature_row.append(pred_ts.month)
    feature_row.append(1 if pred_ts.dayofweek >= 5 else 0)

    # Make prediction
    X_pred = np.array(feature_row).reshape(1, -1)
    X_pred_scaled = scaler_X.transform(X_pred)
    X_pred_nn = X_pred_scaled.T

    dummy_y = np.zeros((1, 1))
    pred_scaled = predict(X_pred_nn, dummy_y, parameters)
    pred_consumption = scaler_y.inverse_transform(pred_scaled.flatten().reshape(-1, 1)).flatten()[0]

    predictions_raw.append(pred_consumption)
    valid_timestamps.append(pred_ts)

    # Apply variance enhancement based on mode
    if VARIANCE_MODE == 'none':
        pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'residual_boost':
        # Add random noise based on historical residuals at this time
        time_key = pred_ts.time()
        if time_key in hist_by_time.index:
            hist_std = hist_by_time.loc[time_key, 'std']
            residual = np.random.normal(0, hist_std * BLEND_RATIO)
            pred_enhanced = pred_consumption + residual
        else:
            pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'pattern_match':
        # Blend with similar days' pattern
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            pattern_deviation = avg_pattern[i] - avg_pattern.mean()
            pattern_std_at_time = std_pattern[i] if i < len(std_pattern) else std_pattern.mean()
            pred_enhanced = pred_consumption + pattern_deviation * BLEND_RATIO
            if pattern_std_at_time > 0:
                noise = np.random.normal(0, pattern_std_at_time * BLEND_RATIO * 0.5)
                pred_enhanced += noise
        else:
            pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'hybrid':
        # HYBRID: Use historical pattern as BASE, DeepNN provides adjustment
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            hist_value = avg_pattern[i]
            pred_enhanced = (hist_value * BLEND_RATIO +
                           pred_consumption * (1 - BLEND_RATIO))
            pattern_std_at_time = std_pattern[i] if i < len(std_pattern) else std_pattern.mean()
            if pattern_std_at_time > 0:
                noise = np.random.normal(0, pattern_std_at_time * 0.2)
                pred_enhanced += noise
        else:
            pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'volatility_scale':
        # Scale predictions to match historical volatility
        time_key = pred_ts.time()
        if time_key in hist_by_time.index:
            hist_mean = hist_by_time.loc[time_key, 'mean']
            hist_std = hist_by_time.loc[time_key, 'std']
            if hist_std > 0:
                z_score = (pred_consumption - hist_mean) / max(hist_std, 1)
                pred_enhanced = hist_mean + z_score * hist_std * (1 + BLEND_RATIO)
            else:
                pred_enhanced = pred_consumption
        else:
            pred_enhanced = pred_consumption
    else:
        pred_enhanced = pred_consumption

    predictions_enhanced.append(pred_enhanced)

    # Update rolling buffer with enhanced prediction (for autoregressive feedback)
    new_row = pd.DataFrame({
        'DateTime': [pred_ts],
        'Consumption': [pred_enhanced],
        'Temperature': [temp_actual],
        'Temperature_Predicted': [future_temp]
    })

    rolling_buffer = pd.concat([rolling_buffer.iloc[1:], new_row], ignore_index=True)

    # Progress update
    if i < 12 or (i + 1) % 12 == 0:
        hist_val = avg_pattern[i] if len(similar_patterns) > 0 and i < len(avg_pattern) else 0
        print(f"  {pred_ts.strftime('%H:%M')} - Raw DeepNN: {pred_consumption:.1f} kW, "
              f"Hist Pattern: {hist_val:.1f} kW, Enhanced: {pred_enhanced:.1f} kW")

predictions_raw = np.array(predictions_raw)
predictions_enhanced = np.array(predictions_enhanced)

print(f"\nGenerated {len(predictions_enhanced)} consumption forecasts")

# 6 - COMPARE STATISTICS
print("\n" + "="*70)
print("VARIANCE COMPARISON")
print("="*70)

# Get last COMPARISON_DAYS days for comparison
historical_days_data = {}
for d in range(1, COMPARISON_DAYS + 1):
    comp_date = forecast_date - pd.Timedelta(days=d)
    comp_day_data = historical_data[historical_data['DateOnly'] == comp_date].copy()
    comp_day_data = comp_day_data.sort_values('DateTime').reset_index(drop=True)
    if len(comp_day_data) >= 288:
        historical_days_data[comp_date] = comp_day_data['Consumption'].values[:288]
        print(f"  Found data for {comp_date} ({pd.Timestamp(comp_date).strftime('%A')})")

# Previous day for main comparison
prev_date = forecast_date - pd.Timedelta(days=1)
if prev_date in historical_days_data:
    prev_consumption = historical_days_data[prev_date]
else:
    prev_consumption = None

print(f"\n{'Metric':<25} {'Raw DeepNN':>15} {'Enhanced':>15} {'Prev Day':>15}")
print("-" * 70)
print(f"{'Mean (kW)':<25} {predictions_raw.mean():>15.1f} {predictions_enhanced.mean():>15.1f} {prev_consumption.mean() if prev_consumption is not None else 'N/A':>15}")
print(f"{'Std Dev (kW)':<25} {predictions_raw.std():>15.1f} {predictions_enhanced.std():>15.1f} {prev_consumption.std() if prev_consumption is not None else 'N/A':>15}")
print(f"{'Max (kW)':<25} {predictions_raw.max():>15.1f} {predictions_enhanced.max():>15.1f} {prev_consumption.max() if prev_consumption is not None else 'N/A':>15}")
print(f"{'Min (kW)':<25} {predictions_raw.min():>15.1f} {predictions_enhanced.min():>15.1f} {prev_consumption.min() if prev_consumption is not None else 'N/A':>15}")

# 7 - SAVE RESULTS
results_df = pd.DataFrame({
    'DateTime': valid_timestamps,
    'Time': [ts.strftime('%H:%M') for ts in valid_timestamps],
    'Raw_Prediction_kW': predictions_raw,
    'Enhanced_Prediction_kW': predictions_enhanced
})

results_filename = f'Cons_DeepNN_Forecast_Enhanced_{FORECAST_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nForecast saved to: {results_filename}")

# 8 - PLOT COMPARISON WITH LAST 4 DAYS
print("\n" + "="*70)
print(f"GENERATING COMPARISON PLOTS WITH LAST {COMPARISON_DAYS} DAYS")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(18, 14))

x_vals = np.arange(len(valid_timestamps))

# Color palette for historical days (from light to dark)
hist_colors = ['#a8d5a2', '#7bc47f', '#4db35a', '#2e8b40']  # Light to dark green

# Plot 1: All predictions comparison with last 4 days
ax1 = axes[0]

# Plot historical days (from oldest to newest)
sorted_hist_dates = sorted(historical_days_data.keys())
for idx, hist_date in enumerate(sorted_hist_dates):
    hist_consumption = historical_days_data[hist_date]
    if len(hist_consumption) == len(predictions_enhanced):
        day_name = day_names[pd.Timestamp(hist_date).dayofweek]
        days_ago = (forecast_date - hist_date).days
        color = hist_colors[min(idx, len(hist_colors)-1)]
        alpha = 0.4 + 0.15 * idx
        linewidth = 1.0 + 0.3 * idx
        ax1.plot(x_vals, hist_consumption, '-', color=color,
                label=f'{hist_date} ({day_name}, -{days_ago}d)',
                linewidth=linewidth, alpha=alpha)

# Plot raw DeepNN forecast
ax1.plot(x_vals, predictions_raw, 'b--', label='Raw DeepNN Forecast', linewidth=1.5, alpha=0.6)

# Plot enhanced forecast (most prominent)
ax1.plot(x_vals, predictions_enhanced, 'r-', label=f'Enhanced Forecast ({VARIANCE_MODE})',
         linewidth=3, alpha=0.9)

# Set Y-axis scale based on all data
all_values = list(predictions_enhanced) + list(predictions_raw)
for hist_consumption in historical_days_data.values():
    all_values.extend(hist_consumption)
y_min = min(all_values) * 0.9
y_max = max(all_values) * 1.05
ax1.set_ylim(y_min, y_max)

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Consumption (kW)', fontsize=12)
ax1.set_title(f'Enhanced DeepNN Forecast for {FORECAST_DATE} vs Last {len(historical_days_data)} Days\n'
             f'Mode: {VARIANCE_MODE}, Blend Ratio: {BLEND_RATIO}',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)

# X-axis labels every 2 hours
tick_positions = np.arange(0, len(valid_timestamps), 24)
tick_labels = [valid_timestamps[i].strftime('%H:%M') for i in tick_positions if i < len(valid_timestamps)]
ax1.set_xticks(tick_positions[:len(tick_labels)])
ax1.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)

ax1.minorticks_on()
ax1.grid(which='minor', alpha=0.15)

# Plot 2: Forecast vs Average of last days
ax2 = axes[1]

if len(historical_days_data) > 0:
    hist_arrays = [v for v in historical_days_data.values() if len(v) == len(predictions_enhanced)]
    if hist_arrays:
        hist_avg = np.mean(hist_arrays, axis=0)
        hist_std = np.std(hist_arrays, axis=0)

        # Plot confidence band
        ax2.fill_between(x_vals, hist_avg - hist_std, hist_avg + hist_std,
                        color='green', alpha=0.2, label='Historical +/- 1 std')
        ax2.plot(x_vals, hist_avg, 'g-', label=f'Avg of last {len(hist_arrays)} days',
                linewidth=2, alpha=0.8)

# Plot forecasts
ax2.plot(x_vals, predictions_raw, 'b--', label='Raw DeepNN', linewidth=1.5, alpha=0.6)
ax2.plot(x_vals, predictions_enhanced, 'r-', label='Enhanced Forecast', linewidth=2.5)

ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_ylabel('Consumption (kW)', fontsize=12)
ax2.set_title('Forecast vs Historical Average (with confidence band)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(tick_positions[:len(tick_labels)])
ax2.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)
ax2.minorticks_on()
ax2.grid(which='minor', alpha=0.15)

plt.tight_layout()
plot_filename = f'Cons_DeepNN_Forecast_Enhanced_{FORECAST_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_filename}")
plt.show()

# 9 - SUMMARY
print("\n" + "#"*70)
print("### ENHANCED DEEPNN FORECAST SUMMARY")
print("#"*70)

print(f"\nForecast Date: {FORECAST_DATE}")
print(f"Variance Mode: {VARIANCE_MODE}")
print(f"Blend Ratio: {BLEND_RATIO}")
print(f"Intervals Forecasted: {len(predictions_enhanced)}")

print(f"\n--- Enhanced Forecast Statistics ---")
print(f"Total Predicted Consumption: {predictions_enhanced.sum() / 12:.2f} kWh")
print(f"Average Predicted Power:     {predictions_enhanced.mean():.2f} kW")
print(f"Standard Deviation:          {predictions_enhanced.std():.2f} kW")
print(f"Peak Predicted Power:        {predictions_enhanced.max():.2f} kW at {valid_timestamps[predictions_enhanced.argmax()].strftime('%H:%M')}")
print(f"Min Predicted Power:         {predictions_enhanced.min():.2f} kW at {valid_timestamps[predictions_enhanced.argmin()].strftime('%H:%M')}")

if len(historical_days_data) > 0:
    print(f"\n--- Comparison with Last {len(historical_days_data)} Days ---")
    print(f"{'Date':<12} {'Day':<6} {'Total kWh':>12} {'Mean kW':>10} {'Std kW':>10} {'Corr Raw':>10} {'Corr Enh':>10}")
    print("-" * 75)

    for hist_date in sorted(historical_days_data.keys(), reverse=True):
        hist_consumption = historical_days_data[hist_date]
        if len(hist_consumption) == len(predictions_enhanced):
            day_name = day_names[pd.Timestamp(hist_date).dayofweek][:3]
            total_kwh = hist_consumption.sum() / 12
            mean_kw = hist_consumption.mean()
            std_kw = hist_consumption.std()
            corr_raw = np.corrcoef(predictions_raw, hist_consumption)[0, 1]
            corr_enh = np.corrcoef(predictions_enhanced, hist_consumption)[0, 1]
            print(f"{str(hist_date):<12} {day_name:<6} {total_kwh:>12.1f} {mean_kw:>10.1f} {std_kw:>10.1f} {corr_raw:>10.4f} {corr_enh:>10.4f}")

    # Average statistics
    hist_arrays = [v for v in historical_days_data.values() if len(v) == len(predictions_enhanced)]
    if hist_arrays:
        avg_consumption = np.mean(hist_arrays, axis=0)
        print("-" * 75)
        print(f"{'AVERAGE':<12} {'':<6} {avg_consumption.sum()/12:>12.1f} {avg_consumption.mean():>10.1f} {np.mean([np.std(h) for h in hist_arrays]):>10.1f}")
        print(f"{'FORECAST':<12} {'':<6} {predictions_enhanced.sum()/12:>12.1f} {predictions_enhanced.mean():>10.1f} {predictions_enhanced.std():>10.1f}")

print(f"\n--- Model Info ---")
print(f"Model file: {MODEL_FILE}")
print(f"Architecture: {LAYERS_DIMS}")
print(f"Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
print(f"Training R²: {float(model_data['training_r2']):.4f}")

print("\n" + "#"*70)
print("### ENHANCED DEEPNN FORECAST COMPLETE!")
print("#"*70)
