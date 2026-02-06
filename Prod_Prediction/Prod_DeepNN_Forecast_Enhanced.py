"""
Enhanced DeepNN Production Forecasting with Variance Preservation
=================================================================
This script loads a pre-trained DeepNN model and makes production forecasts
for a future date, using variance enhancement techniques.

PRODUCTION FORECASTING SPECIFICS:
---------------------------------
Solar/Hydro production has different patterns than consumption:
1. Solar: Strongly correlated with irradiance, peaks at midday
2. Hydro: Depends on water levels, more stable baseline
3. Combined: Pattern follows sun cycle + water availability

VARIANCE ENHANCEMENT MODES:
---------------------------
- 'none': Raw DeepNN predictions (may be smooth)
- 'hybrid': Blend historical pattern with DeepNN (recommended)
- 'pattern_match': Use similar days' patterns
- 'irradiance_weighted': Weight by irradiance correlation

Based on research: https://www.nature.com/articles/s41598-025-14908-x
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
FORECAST_DATE = '2025-01-28'              # Date to forecast
MODEL_FILE = 'Prod_DeepNN_Model.npz'      # Saved model file
DATA_FILE = 'matis_2025_.csv'             # Data file

# ENHANCEMENT OPTIONS
VARIANCE_MODE = 'hybrid'         # 'none', 'hybrid', 'pattern_match', 'irradiance_weighted'
BLEND_RATIO = 0.7                # For 'hybrid': 0.7 = 70% history, 30% DeepNN
SIMILAR_DAYS_COUNT = 5           # Number of similar days for pattern matching
COMPARISON_DAYS = 4              # Days to show in comparison plot
# ========================================================================

print("="*70)
print("ENHANCED DeepNN PRODUCTION FORECASTING")
print(f"Forecast date: {FORECAST_DATE}")
print(f"Variance preservation mode: {VARIANCE_MODE}")
print("="*70)

# 1 - LOAD THE TRAINED MODEL
print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found!\n"
                           f"Please run Prod_DeepNN_Prediction.py with SAVE_MODEL=True first.")

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

data = pd.read_csv(DATA_FILE)

# Parse DateTime
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Daytime'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
data = data.dropna(subset=['DateTime'])
data = data.sort_values('DateTime').reset_index(drop=True)

# Rename columns
data.rename(columns={
    'Haselholtz Water level': 'Level_Haselholz',
    'Bidmi Water level': 'Level_Bidmi'
}, inplace=True)

# Add date column
data['Date_only'] = data['DateTime'].dt.date

print(f"Data loaded: {len(data)} rows")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

# 3 - PREPARE DATA
print("\n" + "="*70)
print(f"PREPARING FORECAST FOR {FORECAST_DATE}")
print("="*70)

forecast_date = pd.to_datetime(FORECAST_DATE).date()

# Get data BEFORE the forecast date
historical_data = data[data['Date_only'] < forecast_date].copy()
print(f"Historical data available: {len(historical_data)} rows")

# Get data for the forecast date (for irradiance forecast if available)
forecast_date_data = data[data['Date_only'] == forecast_date].copy()
if len(forecast_date_data) > 0:
    print(f"Forecast date data found: {len(forecast_date_data)} values")
else:
    print(f"No data found for {FORECAST_DATE} - will use last known values")

# 4 - FIND SIMILAR HISTORICAL DAYS
print("\n" + "="*70)
print("FINDING SIMILAR HISTORICAL DAYS")
print("="*70)

def find_similar_days(historical_data, target_date, n_similar=5):
    """
    Find days similar to target date based on:
    - Day of year (seasonal pattern)
    - Average irradiance pattern
    """
    target_doy = pd.Timestamp(target_date).dayofyear

    historical_dates = historical_data['Date_only'].unique()

    similarities = []
    for hist_date in historical_dates:
        hist_doy = pd.Timestamp(hist_date).dayofyear

        day_data = historical_data[historical_data['Date_only'] == hist_date]
        if len(day_data) < 200:
            continue

        # Day of year similarity (closer = better)
        doy_diff = min(abs(target_doy - hist_doy), 365 - abs(target_doy - hist_doy))

        # Average irradiance (proxy for weather similarity)
        avg_irradiance = day_data['Irradiance'].mean()

        score = doy_diff  # Simple: just use day of year proximity

        similarities.append({
            'date': hist_date,
            'score': score,
            'doy': hist_doy,
            'avg_irradiance': avg_irradiance,
            'production_mean': day_data['Production'].mean(),
            'production_std': day_data['Production'].std(),
            'production_max': day_data['Production'].max()
        })

    similarities.sort(key=lambda x: x['score'])
    return similarities[:n_similar]

similar_days = find_similar_days(historical_data, forecast_date, SIMILAR_DAYS_COUNT)
print(f"\nTop {SIMILAR_DAYS_COUNT} similar days to {FORECAST_DATE}:")
for i, day in enumerate(similar_days):
    print(f"  {i+1}. {day['date']} (DoY: {day['doy']}) - "
          f"Mean: {day['production_mean']:.0f} kW, Max: {day['production_max']:.0f} kW, "
          f"Irr: {day['avg_irradiance']:.1f} W/m²")

# Get production patterns from similar days
similar_patterns = []
similar_irradiance = []
for day in similar_days:
    day_data = historical_data[historical_data['Date_only'] == day['date']].copy()
    day_data = day_data.sort_values('DateTime').reset_index(drop=True)
    if len(day_data) == 288:
        similar_patterns.append(day_data['Production'].values)
        similar_irradiance.append(day_data['Irradiance'].values)

if similar_patterns:
    similar_patterns = np.array(similar_patterns)
    similar_irradiance = np.array(similar_irradiance)
    avg_pattern = np.mean(similar_patterns, axis=0)
    std_pattern = np.std(similar_patterns, axis=0)
    avg_irradiance_pattern = np.mean(similar_irradiance, axis=0)
    print(f"\nSimilar days pattern - Mean: {avg_pattern.mean():.0f} kW, Max: {avg_pattern.max():.0f} kW")

# 5 - AUTOREGRESSIVE FORECASTING
print("\n" + "="*70)
print("AUTOREGRESSIVE FORECAST WITH VARIANCE ENHANCEMENT")
print(f"Mode: {VARIANCE_MODE}")
print(f"Blend Ratio: {BLEND_RATIO}")
print("="*70)

# Generate timestamps for forecast day
forecast_start = pd.Timestamp(f"{FORECAST_DATE} 00:00:00")
forecast_end = pd.Timestamp(f"{FORECAST_DATE} 23:55:00")
forecast_timestamps = pd.date_range(start=forecast_start, end=forecast_end, freq='5min')

print(f"Generating forecasts for {len(forecast_timestamps)} time intervals")

# Build rolling buffer from historical data
rolling_buffer = historical_data.tail(LOOKBACK_STEPS).copy().reset_index(drop=True)

if len(rolling_buffer) < LOOKBACK_STEPS:
    raise ValueError(f"Insufficient historical data: {len(rolling_buffer)}/{LOOKBACK_STEPS}")

# Get historical stats by time
hist_by_time = historical_data.groupby(historical_data['DateTime'].dt.time)['Production'].agg(['mean', 'std'])

predictions_raw = []
predictions_enhanced = []
valid_timestamps = []

print(f"\nStarting autoregressive forecasting...")
print(f"Initial buffer - Last production values: {rolling_buffer['Production'].tail(5).values}")
print(f"Initial buffer - Mean production: {rolling_buffer['Production'].mean():.1f} kW")

for i, pred_ts in enumerate(forecast_timestamps):
    # Get irradiance for this timestamp (use forecast if available, else historical pattern)
    if len(forecast_date_data) > 0:
        irr_row = forecast_date_data[forecast_date_data['DateTime'] == pred_ts]
        if len(irr_row) > 0:
            future_irradiance = irr_row['Irradiance'].values[0]
        else:
            # Use similar days' irradiance pattern
            future_irradiance = avg_irradiance_pattern[i] if i < len(avg_irradiance_pattern) else 0
    else:
        future_irradiance = avg_irradiance_pattern[i] if len(similar_patterns) > 0 and i < len(avg_irradiance_pattern) else 0

    # Build feature vector
    feature_row = []

    # Past production
    past_production = rolling_buffer['Production'].values
    feature_row.extend(past_production)

    # Past irradiance
    past_irradiance = rolling_buffer['Irradiance'].values
    feature_row.extend(past_irradiance)

    # Past temperature
    past_temperature = rolling_buffer['Temperature'].values
    feature_row.extend(past_temperature)

    # Past water levels
    past_level_bidmi = rolling_buffer['Level_Bidmi'].values
    feature_row.extend(past_level_bidmi)

    past_level_haselholz = rolling_buffer['Level_Haselholz'].values
    feature_row.extend(past_level_haselholz)

    # Future irradiance
    feature_row.append(future_irradiance)

    # Time features
    feature_row.append(np.sin(2 * np.pi * pred_ts.hour / 24))
    feature_row.append(np.cos(2 * np.pi * pred_ts.hour / 24))
    feature_row.append(np.sin(2 * np.pi * pred_ts.dayofyear / 365))
    feature_row.append(np.cos(2 * np.pi * pred_ts.dayofyear / 365))
    feature_row.append(pred_ts.month)

    # Normalize and predict
    X_forecast = np.array(feature_row).reshape(1, -1)
    X_forecast_scaled = scaler_X.transform(X_forecast)
    X_forecast_nn = X_forecast_scaled.T

    dummy_y = np.zeros((1, 1))
    pred_scaled = predict(X_forecast_nn, dummy_y, parameters)
    pred_production = scaler_y.inverse_transform(pred_scaled.flatten().reshape(-1, 1)).flatten()[0]

    # Ensure non-negative
    pred_production = max(pred_production, 0)
    predictions_raw.append(pred_production)

    # VARIANCE ENHANCEMENT
    if VARIANCE_MODE == 'none':
        pred_enhanced = pred_production

    elif VARIANCE_MODE == 'hybrid':
        # Blend historical pattern with DeepNN prediction
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            hist_value = avg_pattern[i]
            pred_enhanced = (hist_value * BLEND_RATIO + pred_production * (1 - BLEND_RATIO))

            # Add noise based on historical variance
            if i < len(std_pattern) and std_pattern[i] > 0:
                noise = np.random.normal(0, std_pattern[i] * 0.15)
                pred_enhanced += noise
        else:
            pred_enhanced = pred_production

    elif VARIANCE_MODE == 'pattern_match':
        # Use historical pattern with DeepNN adjustment
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            pattern_deviation = avg_pattern[i] - avg_pattern.mean()
            pred_enhanced = pred_production + pattern_deviation * BLEND_RATIO
        else:
            pred_enhanced = pred_production

    elif VARIANCE_MODE == 'irradiance_weighted':
        # Weight by irradiance correlation
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            hist_value = avg_pattern[i]

            # Scale based on irradiance ratio
            if i < len(avg_irradiance_pattern) and avg_irradiance_pattern[i] > 0:
                irr_ratio = future_irradiance / (avg_irradiance_pattern[i] + 1e-6)
                irr_ratio = np.clip(irr_ratio, 0.5, 2.0)  # Limit extreme values
            else:
                irr_ratio = 1.0

            pred_enhanced = hist_value * irr_ratio * BLEND_RATIO + pred_production * (1 - BLEND_RATIO)
        else:
            pred_enhanced = pred_production
    else:
        pred_enhanced = pred_production

    # Ensure non-negative
    pred_enhanced = max(pred_enhanced, 0)
    predictions_enhanced.append(pred_enhanced)
    valid_timestamps.append(pred_ts)

    # Update rolling buffer
    new_row = pd.DataFrame({
        'DateTime': [pred_ts],
        'Production': [pred_enhanced],
        'Irradiance': [future_irradiance],
        'Temperature': [rolling_buffer['Temperature'].iloc[-1]],  # Use last known
        'Level_Bidmi': [rolling_buffer['Level_Bidmi'].iloc[-1]],
        'Level_Haselholz': [rolling_buffer['Level_Haselholz'].iloc[-1]]
    })

    rolling_buffer = pd.concat([rolling_buffer.iloc[1:], new_row], ignore_index=True)

    # Progress update
    if (i + 1) % 12 == 0:
        hist_val = avg_pattern[i] if len(similar_patterns) > 0 and i < len(avg_pattern) else 0
        print(f"  {pred_ts.strftime('%H:%M')} - Raw: {pred_production:.1f} kW, "
              f"Hist: {hist_val:.1f} kW, Enhanced: {pred_enhanced:.1f} kW")

predictions_raw = np.array(predictions_raw)
predictions_enhanced = np.array(predictions_enhanced)

print(f"\nGenerated {len(predictions_enhanced)} production forecasts")

# 6 - COMPARE STATISTICS
print("\n" + "="*70)
print("VARIANCE COMPARISON")
print("="*70)

# Get last N days for comparison
historical_days_data = {}
for d in range(1, COMPARISON_DAYS + 1):
    comp_date = forecast_date - pd.Timedelta(days=d)
    comp_day_data = historical_data[historical_data['Date_only'] == comp_date].copy()
    comp_day_data = comp_day_data.sort_values('DateTime').reset_index(drop=True)
    if len(comp_day_data) == 288:
        historical_days_data[comp_date] = comp_day_data['Production'].values
        print(f"  Found data for {comp_date}")

prev_date = forecast_date - pd.Timedelta(days=1)
if prev_date in historical_days_data:
    prev_production = historical_days_data[prev_date]
else:
    prev_production = None

print(f"\n{'Metric':<25} {'Raw DeepNN':>15} {'Enhanced':>15} {'Prev Day':>15}")
print("-" * 70)
print(f"{'Mean (kW)':<25} {predictions_raw.mean():>15.1f} {predictions_enhanced.mean():>15.1f} {prev_production.mean() if prev_production is not None else 'N/A':>15}")
print(f"{'Std Dev (kW)':<25} {predictions_raw.std():>15.1f} {predictions_enhanced.std():>15.1f} {prev_production.std() if prev_production is not None else 'N/A':>15}")
print(f"{'Max (kW)':<25} {predictions_raw.max():>15.1f} {predictions_enhanced.max():>15.1f} {prev_production.max() if prev_production is not None else 'N/A':>15}")
print(f"{'Min (kW)':<25} {predictions_raw.min():>15.1f} {predictions_enhanced.min():>15.1f} {prev_production.min() if prev_production is not None else 'N/A':>15}")

# 7 - CREATE RESULTS
results_df = pd.DataFrame({
    'DateTime': valid_timestamps,
    'Time': [ts.strftime('%H:%M') for ts in valid_timestamps],
    'Raw_Prediction_kW': predictions_raw,
    'Enhanced_Prediction_kW': predictions_enhanced
})

results_filename = f'Prod_DeepNN_Forecast_Enhanced_{FORECAST_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nForecast saved to: {results_filename}")

# 8 - PLOT COMPARISON
print("\n" + "="*70)
print("GENERATING COMPARISON PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(18, 14))

x_vals = np.arange(len(valid_timestamps))

hist_colors = ['#a8d5a2', '#7bc47f', '#4db35a', '#2e8b40']
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Plot 1: All predictions comparison with historical days
ax1 = axes[0]

sorted_hist_dates = sorted(historical_days_data.keys())
for idx, hist_date in enumerate(sorted_hist_dates):
    hist_production = historical_days_data[hist_date]
    if len(hist_production) == len(predictions_enhanced):
        day_name = day_names[pd.Timestamp(hist_date).dayofweek]
        days_ago = (forecast_date - hist_date).days
        color = hist_colors[min(idx, len(hist_colors)-1)]
        alpha = 0.4 + 0.15 * idx
        linewidth = 1.0 + 0.3 * idx
        ax1.plot(x_vals, hist_production, '-', color=color,
                label=f'{hist_date} ({day_name}, -{days_ago}d)',
                linewidth=linewidth, alpha=alpha)

ax1.plot(x_vals, predictions_raw, 'b--', label='Raw DeepNN', linewidth=1.5, alpha=0.6)
ax1.plot(x_vals, predictions_enhanced, 'r-', label=f'Enhanced ({VARIANCE_MODE})',
         linewidth=3, alpha=0.9)

all_values = list(predictions_enhanced) + list(predictions_raw)
for hist_production in historical_days_data.values():
    all_values.extend(hist_production)
y_min = max(0, min(all_values) * 0.9)
y_max = max(all_values) * 1.05
ax1.set_ylim(y_min, y_max)

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Production (kW)', fontsize=12)
ax1.set_title(f'Enhanced DeepNN Production Forecast for {FORECAST_DATE} vs Last {len(historical_days_data)} Days\n'
             f'Mode: {VARIANCE_MODE}, Blend Ratio: {BLEND_RATIO}',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)

tick_positions = np.arange(0, len(valid_timestamps), 24)
tick_labels = [valid_timestamps[i].strftime('%H:%M') for i in tick_positions if i < len(valid_timestamps)]
ax1.set_xticks(tick_positions[:len(tick_labels)])
ax1.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)
ax1.minorticks_on()
ax1.grid(which='minor', alpha=0.15)

# Plot 2: Forecast vs Historical Average with irradiance
ax2 = axes[1]

if len(historical_days_data) > 0:
    hist_arrays = [v for v in historical_days_data.values() if len(v) == len(predictions_enhanced)]
    if hist_arrays:
        hist_avg = np.mean(hist_arrays, axis=0)
        hist_std = np.std(hist_arrays, axis=0)

        ax2.fill_between(x_vals, hist_avg - hist_std, hist_avg + hist_std,
                        color='green', alpha=0.2, label='Historical ±1 std')
        ax2.plot(x_vals, hist_avg, 'g-', label=f'Avg of last {len(hist_arrays)} days',
                linewidth=2, alpha=0.8)

ax2.plot(x_vals, predictions_raw, 'b--', label='Raw DeepNN', linewidth=1.5, alpha=0.6)
ax2.plot(x_vals, predictions_enhanced, 'r-', label='Enhanced Forecast', linewidth=2.5)

ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_ylabel('Production (kW)', fontsize=12)
ax2.set_title('Forecast vs Historical Average (with confidence band)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(tick_positions[:len(tick_labels)])
ax2.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)
ax2.minorticks_on()
ax2.grid(which='minor', alpha=0.15)

plt.tight_layout()
plot_filename = f'Prod_DeepNN_Forecast_Enhanced_{FORECAST_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_filename}")
plt.show()

# 9 - SUMMARY
print("\n" + "#"*70)
print("### ENHANCED DeepNN PRODUCTION FORECAST SUMMARY")
print("#"*70)

print(f"\nForecast Date: {FORECAST_DATE}")
print(f"Enhancement Mode: {VARIANCE_MODE}")
print(f"Blend Ratio: {BLEND_RATIO}")

print(f"\n--- Enhanced Forecast Statistics ---")
print(f"Total Predicted Production: {predictions_enhanced.sum() / 12:.2f} kWh")
print(f"Average Predicted Power:    {predictions_enhanced.mean():.2f} kW")
print(f"Standard Deviation:         {predictions_enhanced.std():.2f} kW")
print(f"Peak Predicted Power:       {predictions_enhanced.max():.2f} kW at {valid_timestamps[predictions_enhanced.argmax()].strftime('%H:%M')}")
print(f"Min Predicted Power:        {predictions_enhanced.min():.2f} kW at {valid_timestamps[predictions_enhanced.argmin()].strftime('%H:%M')}")

if len(historical_days_data) > 0:
    print(f"\n--- Comparison with Last {len(historical_days_data)} Days ---")
    print(f"{'Date':<12} {'Day':<6} {'Total kWh':>12} {'Mean kW':>10} {'Max kW':>10}")
    print("-" * 55)

    for hist_date in sorted(historical_days_data.keys(), reverse=True):
        hist_production = historical_days_data[hist_date]
        if len(hist_production) == len(predictions_enhanced):
            day_name = day_names[pd.Timestamp(hist_date).dayofweek]
            total_kwh = hist_production.sum() / 12
            mean_kw = hist_production.mean()
            max_kw = hist_production.max()
            print(f"{str(hist_date):<12} {day_name:<6} {total_kwh:>12.1f} {mean_kw:>10.1f} {max_kw:>10.1f}")

    hist_arrays = [v for v in historical_days_data.values() if len(v) == len(predictions_enhanced)]
    if hist_arrays:
        avg_production = np.mean(hist_arrays, axis=0)
        print("-" * 55)
        print(f"{'AVERAGE':<12} {'':<6} {avg_production.sum()/12:>12.1f} {avg_production.mean():>10.1f} {avg_production.max():>10.1f}")
        print(f"{'FORECAST':<12} {'':<6} {predictions_enhanced.sum()/12:>12.1f} {predictions_enhanced.mean():>10.1f} {predictions_enhanced.max():>10.1f}")

print("\n" + "#"*70)
print("### ENHANCED PRODUCTION FORECAST COMPLETE!")
print("#"*70)
