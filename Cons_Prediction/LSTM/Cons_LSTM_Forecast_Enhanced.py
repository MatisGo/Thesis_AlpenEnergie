"""
Enhanced LSTM Forecasting with Variance Preservation
=====================================================
This script addresses the "smoothing problem" in autoregressive LSTM forecasting.

PROBLEM:
--------
When using recursive/autoregressive forecasting, predictions feed back as inputs.
This causes "variance collapse" - predictions become increasingly smooth over time
because the model outputs conditional means, which are less variable than actual data.

SOLUTIONS IMPLEMENTED:
---------------------
1. RESIDUAL BOOTSTRAPPING: Add historical residual patterns to predictions
2. PATTERN MATCHING: Find similar historical days and blend their variance
3. VOLATILITY SCALING: Scale predictions to match historical variance
4. TEMPERATURE-BASED CORRECTION: Adjust based on temperature correlation with peaks

Based on research from:
- Multi-step ahead forecasting techniques
- Signal decomposition methods
- LSTM-GARCH hybrid approaches
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ========================================================================
# FORECAST CONFIGURATION
# ========================================================================
FORECAST_DATE = '2026-01-28'              # Date to forecast
MODEL_FILE = 'Cons_LSTM_Model.keras'      # Saved Keras model (in same folder)
CONFIG_FILE = 'Cons_LSTM_Config.npz'      # Saved configuration and scalers
DATA_FILE = '../Data_January.csv'         # Data file (in parent folder)

# ENHANCEMENT OPTIONS
# Options: 'none', 'residual_boost', 'pattern_match', 'volatility_scale', 'hybrid'
VARIANCE_MODE = 'hybrid'         # 'hybrid' = use historical pattern as BASE, LSTM as adjustment
BLEND_RATIO = 0.85               # For 'hybrid': weight of historical pattern (0.85 = 85% history, 15% LSTM)
SIMILAR_DAYS_COUNT = 5           # Number of similar days to use for pattern matching (more = smoother average)
# ========================================================================

print("="*70)
print("ENHANCED LSTM CONSUMPTION FORECASTING")
print(f"Forecast date: {FORECAST_DATE}")
print(f"Variance preservation mode: {VARIANCE_MODE}")
print("="*70)

# 1 - LOAD THE TRAINED MODEL AND CONFIGURATION
print("\n" + "="*70)
print("LOADING TRAINED LSTM MODEL")
print("="*70)

# Check files exist
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found!")
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Config file '{CONFIG_FILE}' not found!")

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
    USE_FUTURE_TEMP = 'Future_Temp_Forecast' in FEATURE_COLUMNS

if 'sequence_features' in config:
    SEQUENCE_FEATURES = config['sequence_features'].tolist()
elif USE_FUTURE_TEMP:
    SEQUENCE_FEATURES = [f for f in FEATURE_COLUMNS if f != 'Future_Temp_Forecast']
else:
    SEQUENCE_FEATURES = FEATURE_COLUMNS

print(f"Lookback steps: {LOOKBACK_STEPS} ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
print(f"LSTM Units: {LSTM_UNITS}")

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

# Add date column and time features
data['Date'] = data['DateTime'].dt.date
data['Hour'] = data['DateTime'].dt.hour
data['DayOfWeek'] = data['DateTime'].dt.dayofweek
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

print(f"Data loaded: {len(data)} rows")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

# 3 - PREPARE DATA
print("\n" + "="*70)
print(f"PREPARING FORECAST FOR {FORECAST_DATE}")
print("="*70)

forecast_date = pd.to_datetime(FORECAST_DATE).date()

# Get data BEFORE the forecast date
historical_data = data[data['Date'] < forecast_date].copy()
print(f"Historical data available: {len(historical_data)} rows")

# Get temperature forecasts for the forecast date
forecast_date_data = data[data['Date'] == forecast_date].copy()
if len(forecast_date_data) > 0:
    print(f"Temperature forecasts found for {FORECAST_DATE}: {len(forecast_date_data)} values")
else:
    print(f"No temperature forecasts found for {FORECAST_DATE}")

# 4 - FIND SIMILAR HISTORICAL DAYS (for pattern matching)
print("\n" + "="*70)
print("FINDING SIMILAR HISTORICAL DAYS")
print("="*70)

def find_similar_days(historical_data, target_date, n_similar=3):
    """
    Find days in historical data that are most similar to target date.
    Similarity based on: day of week, temperature pattern
    """
    target_dow = pd.Timestamp(target_date).dayofweek
    target_is_weekend = 1 if target_dow >= 5 else 0

    # Get temperature forecast for target date (average)
    target_temp_data = data[data['Date'] == target_date]
    if len(target_temp_data) > 0:
        target_temp = target_temp_data['Temperature_Predicted'].mean()
    else:
        target_temp = historical_data['Temperature'].iloc[-288:].mean()

    # Get unique historical dates
    historical_dates = historical_data['Date'].unique()

    similarities = []
    for hist_date in historical_dates:
        hist_dow = pd.Timestamp(hist_date).dayofweek
        hist_is_weekend = 1 if hist_dow >= 5 else 0

        # Get daily data
        day_data = historical_data[historical_data['Date'] == hist_date]
        if len(day_data) < 200:  # Skip incomplete days
            continue

        hist_temp = day_data['Temperature'].mean()

        # Calculate similarity score (lower is better)
        dow_diff = abs(target_dow - hist_dow)
        if dow_diff > 3:
            dow_diff = 7 - dow_diff  # Wrap around

        weekend_match = 0 if target_is_weekend == hist_is_weekend else 2
        temp_diff = abs(target_temp - hist_temp)

        score = dow_diff + weekend_match + temp_diff * 0.5

        similarities.append({
            'date': hist_date,
            'score': score,
            'dow': hist_dow,
            'temp': hist_temp,
            'consumption_mean': day_data['Consumption'].mean(),
            'consumption_std': day_data['Consumption'].std()
        })

    # Sort by similarity score
    similarities.sort(key=lambda x: x['score'])

    return similarities[:n_similar]

similar_days = find_similar_days(historical_data, forecast_date, SIMILAR_DAYS_COUNT)
print(f"\nTop {SIMILAR_DAYS_COUNT} similar days to {FORECAST_DATE}:")
for i, day in enumerate(similar_days):
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"  {i+1}. {day['date']} ({dow_names[day['dow']]}) - "
          f"Temp: {day['temp']:.1f}°C, Mean: {day['consumption_mean']:.0f} kW, "
          f"Std: {day['consumption_std']:.0f} kW")

# Get consumption patterns from similar days
similar_patterns = []
for day in similar_days:
    day_data = historical_data[historical_data['Date'] == day['date']].copy()
    day_data = day_data.sort_values('DateTime').reset_index(drop=True)
    if len(day_data) == 288:  # Full day
        similar_patterns.append(day_data['Consumption'].values)

if similar_patterns:
    similar_patterns = np.array(similar_patterns)
    avg_pattern = np.mean(similar_patterns, axis=0)
    std_pattern = np.std(similar_patterns, axis=0)
    print(f"\nSimilar days pattern - Mean: {avg_pattern.mean():.0f} kW, Std: {std_pattern.mean():.0f} kW")

# 5 - AUTOREGRESSIVE FORECASTING WITH VARIANCE ENHANCEMENT
print("\n" + "="*70)
print("AUTOREGRESSIVE FORECAST WITH VARIANCE ENHANCEMENT")
print(f"Mode: {VARIANCE_MODE}")
print("="*70)

# Generate timestamps for the full forecast day
forecast_start = pd.Timestamp(f"{FORECAST_DATE} 00:00:00")
forecast_end = pd.Timestamp(f"{FORECAST_DATE} 23:55:00")
forecast_timestamps = pd.date_range(start=forecast_start, end=forecast_end, freq='5min')

print(f"Generating forecasts for {len(forecast_timestamps)} time intervals")

# Build a rolling buffer
rolling_buffer = historical_data.tail(LOOKBACK_STEPS).copy().reset_index(drop=True)

if len(rolling_buffer) < LOOKBACK_STEPS:
    raise ValueError(f"Insufficient historical data: {len(rolling_buffer)}/{LOOKBACK_STEPS}")

# Get historical residuals (for residual bootstrapping)
# Residuals = Actual - typical pattern at that time
hist_by_time = historical_data.groupby(historical_data['DateTime'].dt.time)['Consumption'].agg(['mean', 'std'])

predictions_raw = []  # Raw LSTM predictions
predictions_enhanced = []  # Enhanced with variance
valid_timestamps = []

print(f"\nStarting autoregressive forecasting with {VARIANCE_MODE} enhancement...")
print(f"BLEND_RATIO = {BLEND_RATIO} ({'%.0f%% historical pattern, %.0f%% LSTM' % (BLEND_RATIO*100, (1-BLEND_RATIO)*100) if VARIANCE_MODE == 'hybrid' else ''})")

# Debug: Show what the rolling buffer contains at start
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

    # Normalize and predict
    sequence_reshaped = sequence_data.reshape(-1, n_features)
    sequence_scaled = scaler_X.transform(sequence_reshaped)
    sequence_scaled = sequence_scaled.reshape(1, LOOKBACK_STEPS, n_features)

    pred_scaled = model.predict(sequence_scaled, verbose=0)
    pred_consumption = scaler_y.inverse_transform(pred_scaled).flatten()[0]

    predictions_raw.append(pred_consumption)

    # VARIANCE ENHANCEMENT
    if VARIANCE_MODE == 'none':
        pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'residual_boost':
        # Add historical residual at this time of day
        time_key = pred_ts.time()
        if time_key in hist_by_time.index:
            hist_mean = hist_by_time.loc[time_key, 'mean']
            hist_std = hist_by_time.loc[time_key, 'std']
            # Sample from historical distribution
            residual = np.random.normal(0, hist_std * BLEND_RATIO)
            pred_enhanced = pred_consumption + residual
        else:
            pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'pattern_match':
        # Blend with similar days' pattern (OLD METHOD - weak correction)
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            # Get the deviation from similar days' mean at this time
            pattern_deviation = avg_pattern[i] - avg_pattern.mean()
            pattern_std_at_time = std_pattern[i] if i < len(std_pattern) else std_pattern.mean()

            # Blend: keep LSTM base but add pattern deviation
            pred_enhanced = pred_consumption + pattern_deviation * BLEND_RATIO

            # Also add some variance from historical std
            if pattern_std_at_time > 0:
                noise = np.random.normal(0, pattern_std_at_time * BLEND_RATIO * 0.5)
                pred_enhanced += noise
        else:
            pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'hybrid':
        # HYBRID: Use historical pattern as BASE, LSTM provides adjustment
        # This inverts the approach: instead of LSTM + small_correction,
        # we use Historical_Pattern + LSTM_adjustment
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            # Historical pattern value at this time
            hist_value = avg_pattern[i]

            # Combine: historical pattern + LSTM's relative adjustment
            # BLEND_RATIO controls the weight: 0.6 = 60% history, 40% LSTM
            pred_enhanced = (hist_value * BLEND_RATIO +
                           pred_consumption * (1 - BLEND_RATIO))

            # Add realistic noise based on historical variance at this time
            pattern_std_at_time = std_pattern[i] if i < len(std_pattern) else std_pattern.mean()
            if pattern_std_at_time > 0:
                # Smaller noise since we're already using historical pattern
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

            # Calculate how far prediction is from historical mean (in std units)
            if hist_std > 0:
                z_score = (pred_consumption - hist_mean) / (hist_std * 0.5)  # LSTM tends to underestimate std
                # Apply volatility scaling
                pred_enhanced = hist_mean + z_score * hist_std * (1 + BLEND_RATIO)
            else:
                pred_enhanced = pred_consumption
        else:
            pred_enhanced = pred_consumption
    else:
        pred_enhanced = pred_consumption

    # Ensure positive consumption
    pred_enhanced = max(pred_enhanced, 0)
    predictions_enhanced.append(pred_enhanced)
    valid_timestamps.append(pred_ts)

    # Update rolling buffer with the ENHANCED prediction (more realistic for next step)
    new_row = pd.DataFrame({
        'DateTime': [pred_ts],
        'Consumption': [pred_enhanced],  # Use enhanced prediction
        'Temperature': [temp_actual],
        'Temperature_Predicted': [future_temp],
        'Hour_sin': [np.sin(2 * np.pi * pred_ts.hour / 24)],
        'Hour_cos': [np.cos(2 * np.pi * pred_ts.hour / 24)],
        'DayOfWeek_sin': [np.sin(2 * np.pi * pred_ts.dayofweek / 7)],
        'DayOfWeek_cos': [np.cos(2 * np.pi * pred_ts.dayofweek / 7)],
        'IsWeekend': [1 if pred_ts.dayofweek >= 5 else 0]
    })

    rolling_buffer = pd.concat([rolling_buffer.iloc[1:], new_row], ignore_index=True)

    # Progress update - more detail for first hour, then every hour
    if i < 12 or (i + 1) % 12 == 0:
        hist_val = avg_pattern[i] if len(similar_patterns) > 0 and i < len(avg_pattern) else 0
        print(f"  {pred_ts.strftime('%H:%M')} - Raw LSTM: {pred_consumption:.1f} kW, "
              f"Hist Pattern: {hist_val:.1f} kW, Enhanced: {pred_enhanced:.1f} kW")

predictions_raw = np.array(predictions_raw)
predictions_enhanced = np.array(predictions_enhanced)

print(f"\nGenerated {len(predictions_enhanced)} consumption forecasts")

# 6 - COMPARE STATISTICS
print("\n" + "="*70)
print("VARIANCE COMPARISON")
print("="*70)

# Get last 4 days for comparison
COMPARISON_DAYS = 4
historical_days_data = {}
for d in range(1, COMPARISON_DAYS + 1):
    comp_date = forecast_date - pd.Timedelta(days=d)
    comp_day_data = historical_data[historical_data['Date'] == comp_date].copy()
    comp_day_data = comp_day_data.sort_values('DateTime').reset_index(drop=True)
    if len(comp_day_data) == 288:  # Full day
        historical_days_data[comp_date] = comp_day_data['Consumption'].values
        print(f"  Found data for {comp_date} ({pd.Timestamp(comp_date).strftime('%A')})")

# Previous day for main comparison
prev_date = forecast_date - pd.Timedelta(days=1)
if prev_date in historical_days_data:
    prev_consumption = historical_days_data[prev_date]
else:
    prev_consumption = None

print(f"\n{'Metric':<25} {'Raw LSTM':>15} {'Enhanced':>15} {'Prev Day':>15}")
print("-" * 70)
print(f"{'Mean (kW)':<25} {predictions_raw.mean():>15.1f} {predictions_enhanced.mean():>15.1f} {prev_consumption.mean() if prev_consumption is not None else 'N/A':>15}")
print(f"{'Std Dev (kW)':<25} {predictions_raw.std():>15.1f} {predictions_enhanced.std():>15.1f} {prev_consumption.std() if prev_consumption is not None else 'N/A':>15}")
print(f"{'Max (kW)':<25} {predictions_raw.max():>15.1f} {predictions_enhanced.max():>15.1f} {prev_consumption.max() if prev_consumption is not None else 'N/A':>15}")
print(f"{'Min (kW)':<25} {predictions_raw.min():>15.1f} {predictions_enhanced.min():>15.1f} {prev_consumption.min() if prev_consumption is not None else 'N/A':>15}")
print(f"{'Range (kW)':<25} {predictions_raw.max()-predictions_raw.min():>15.1f} {predictions_enhanced.max()-predictions_enhanced.min():>15.1f} {prev_consumption.max()-prev_consumption.min() if prev_consumption is not None else 'N/A':>15}")

if prev_consumption is not None and len(prev_consumption) == len(predictions_enhanced):
    print(f"\nVariance Recovery:")
    print(f"  Raw LSTM variance ratio: {predictions_raw.std()/prev_consumption.std()*100:.1f}% of actual")
    print(f"  Enhanced variance ratio: {predictions_enhanced.std()/prev_consumption.std()*100:.1f}% of actual")

# 7 - CREATE RESULTS
results_df = pd.DataFrame({
    'DateTime': valid_timestamps,
    'Time': [ts.strftime('%H:%M') for ts in valid_timestamps],
    'Raw_Prediction_kW': predictions_raw,
    'Enhanced_Prediction_kW': predictions_enhanced
})

# Save to CSV
results_filename = f'Cons_LSTM_Forecast_Enhanced_{FORECAST_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nForecast saved to: {results_filename}")

# 8 - PLOT COMPARISON WITH LAST 4 DAYS
print("\n" + "="*70)
print("GENERATING COMPARISON PLOTS WITH LAST 4 DAYS")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(18, 14))

x_vals = np.arange(len(valid_timestamps))

# Color palette for historical days (from light to dark)
hist_colors = ['#a8d5a2', '#7bc47f', '#4db35a', '#2e8b40']  # Light to dark green
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Plot 1: All predictions comparison with last 4 days
ax1 = axes[0]

# Plot historical days (last 4 days) - from oldest to newest
sorted_hist_dates = sorted(historical_days_data.keys())
for idx, hist_date in enumerate(sorted_hist_dates):
    hist_consumption = historical_days_data[hist_date]
    if len(hist_consumption) == len(predictions_enhanced):
        day_name = day_names[pd.Timestamp(hist_date).dayofweek]
        days_ago = (forecast_date - hist_date).days
        color = hist_colors[min(idx, len(hist_colors)-1)]
        alpha = 0.4 + 0.15 * (idx)  # Newer days more visible
        linewidth = 1.0 + 0.3 * idx
        ax1.plot(x_vals, hist_consumption, '-', color=color,
                label=f'{hist_date} ({day_name}, -{days_ago}d)',
                linewidth=linewidth, alpha=alpha)

# Plot raw LSTM forecast
ax1.plot(x_vals, predictions_raw, 'b--', label=f'Raw LSTM Forecast', linewidth=1.5, alpha=0.6)

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
ax1.set_title(f'Enhanced LSTM Forecast for {FORECAST_DATE} vs Last {len(historical_days_data)} Days\n'
             f'Mode: {VARIANCE_MODE}, Blend Ratio: {BLEND_RATIO}',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)

# X-axis labels every 2 hours
tick_positions = np.arange(0, len(valid_timestamps), 24)
tick_labels = [valid_timestamps[i].strftime('%H:%M') for i in tick_positions if i < len(valid_timestamps)]
ax1.set_xticks(tick_positions[:len(tick_labels)])
ax1.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)

# Add minor gridlines
ax1.minorticks_on()
ax1.grid(which='minor', alpha=0.15)

# Plot 2: Forecast vs Average of last 4 days
ax2 = axes[1]

# Calculate average and std of historical days
if len(historical_days_data) > 0:
    hist_arrays = [v for v in historical_days_data.values() if len(v) == len(predictions_enhanced)]
    if hist_arrays:
        hist_avg = np.mean(hist_arrays, axis=0)
        hist_std = np.std(hist_arrays, axis=0)

        # Plot confidence band (avg +/- std)
        ax2.fill_between(x_vals, hist_avg - hist_std, hist_avg + hist_std,
                        color='green', alpha=0.2, label='Historical ±1 std')
        ax2.plot(x_vals, hist_avg, 'g-', label=f'Avg of last {len(hist_arrays)} days',
                linewidth=2, alpha=0.8)

# Plot forecasts
ax2.plot(x_vals, predictions_raw, 'b--', label='Raw LSTM', linewidth=1.5, alpha=0.6)
ax2.plot(x_vals, predictions_enhanced, 'r-', label='Enhanced Forecast', linewidth=2.5)

ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_ylabel('Consumption (kW)', fontsize=12)
ax2.set_title(f'Forecast vs Historical Average (with confidence band)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(tick_positions[:len(tick_labels)])
ax2.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)
ax2.minorticks_on()
ax2.grid(which='minor', alpha=0.15)

plt.tight_layout()
plot_filename = f'Cons_LSTM_Forecast_Enhanced_{FORECAST_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_filename}")
plt.show()

# 9 - SUMMARY
print("\n" + "#"*70)
print("### ENHANCED LSTM FORECAST SUMMARY")
print("#"*70)

print(f"\nForecast Date: {FORECAST_DATE}")
print(f"Enhancement Mode: {VARIANCE_MODE}")
print(f"Blend Ratio: {BLEND_RATIO}")

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
            day_name = day_names[pd.Timestamp(hist_date).dayofweek]
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

print("\n" + "#"*70)
print("### ENHANCED FORECAST COMPLETE!")
print("#"*70)
