"""
LSTM-based Consumption Prediction with Autoregressive Evaluation
================================================================
This script trains an LSTM model and evaluates it using the SAME autoregressive
approach as the Forecast script - meaning predictions feed back into the model.

KEY DIFFERENCE FROM ORIGINAL PREDICTION SCRIPT:
-----------------------------------------------
The original script "cheats" by using actual consumption values in the test data
lookback window. This script uses ONLY data from before the target date, then
makes predictions autoregressively (each prediction feeds into the next).

This gives a REALISTIC evaluation of how the model will perform in production.

WORKFLOW:
1. Train LSTM on data BEFORE target date
2. Make predictions for target date using autoregressive approach
3. Compare with actual values to get TRUE performance metrics
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# TensorFlow/Keras imports for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ========================================================================
# HYPERPARAMETERS - EASY TO CHANGE
# ========================================================================
LEARNING_RATE = 0.001                # Learning rate
EPOCHS = 100                         # Maximum training epochs
BATCH_SIZE = 32                      # Number of samples per gradient update
LOOKBACK_STEPS = 288                 # 288 x 5min = 24 hours of past data
LSTM_UNITS = 64                      # Number of LSTM units
DROPOUT_RATE = 0.2                   # Dropout for regularization
PATIENCE = 10                        # Early stopping patience
SAVE_MODEL = True                    # Save model for real forecasting
# ========================================================================
# VARIANCE ENHANCEMENT OPTIONS (same as Enhanced Forecast)
VARIANCE_MODE = 'pattern_match'      # 'none', 'residual_boost', 'pattern_match', 'volatility_scale'
BLEND_RATIO = 0.3                    # How much historical variance to add
SIMILAR_DAYS_COUNT = 3               # Number of similar days for pattern matching
# ========================================================================
# TARGET DATE TO PREDICT (excluded from training)
TARGET_DATE = '2026-01-27'
# ========================================================================

# 1 - LOAD AND PREPROCESS DATA
print("="*70)
print("LSTM CONSUMPTION PREDICTION WITH AUTOREGRESSIVE EVALUATION")
print(f"Target prediction date: {TARGET_DATE}")
print(f"Variance mode: {VARIANCE_MODE}")
print("="*70)

# Load data from CSV (from parent folder)
data = pd.read_csv('../Data_January.csv', skiprows=3, header=None, encoding='latin-1')
print(f"Loaded data shape: {data.shape}")

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

print(f"Data shape after parsing: {data.shape}")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

# Use 5-minute data directly
data_5min = data.copy()

# Extract time features
data_5min['Hour'] = data_5min['DateTime'].dt.hour
data_5min['Minute'] = data_5min['DateTime'].dt.minute
data_5min['DayOfWeek'] = data_5min['DateTime'].dt.dayofweek
data_5min['Month'] = data_5min['DateTime'].dt.month
data_5min['DayOfYear'] = data_5min['DateTime'].dt.dayofyear
data_5min['IsWeekend'] = (data_5min['DayOfWeek'] >= 5).astype(int)
data_5min['Date'] = data_5min['DateTime'].dt.date

# Create cyclical time features
data_5min['Hour_sin'] = np.sin(2 * np.pi * data_5min['Hour'] / 24)
data_5min['Hour_cos'] = np.cos(2 * np.pi * data_5min['Hour'] / 24)
data_5min['DayOfWeek_sin'] = np.sin(2 * np.pi * data_5min['DayOfWeek'] / 7)
data_5min['DayOfWeek_cos'] = np.cos(2 * np.pi * data_5min['DayOfWeek'] / 7)

# 2 - SPLIT DATA
print("\n" + "="*70)
print(f"SPLITTING DATA - EXCLUDING {TARGET_DATE} FOR PREDICTION")
print("="*70)

target_date = pd.to_datetime(TARGET_DATE).date()

# Training data: all days BEFORE target date
train_data = data_5min[data_5min['Date'] < target_date].copy()

# Test data: target date only (for comparison AFTER prediction)
test_data = data_5min[data_5min['Date'] == target_date].copy()

print(f"Training data: {len(train_data)} intervals (5-min)")
print(f"Test data ({TARGET_DATE}): {len(test_data)} intervals (5-min)")

# 3 - PREPARE FEATURES FOR LSTM
print("\n" + "="*70)
print("PREPARING FEATURES FOR LSTM")
print("="*70)

SEQUENCE_FEATURES = [
    'Consumption',
    'Temperature',
    'Temperature_Predicted',
    'Hour_sin',
    'Hour_cos',
    'DayOfWeek_sin',
    'DayOfWeek_cos',
    'IsWeekend'
]

USE_FUTURE_TEMP = True
FEATURE_COLUMNS = SEQUENCE_FEATURES + (['Future_Temp_Forecast'] if USE_FUTURE_TEMP else [])

print(f"Sequence features: {SEQUENCE_FEATURES}")
print(f"Using future temperature forecast: {USE_FUTURE_TEMP}")

def create_lstm_sequences_with_future_temp(df, sequence_cols, target_col, lookback_steps, use_future_temp=True):
    """Create sequences for LSTM training with optional future temperature forecast."""
    features = df[sequence_cols].values
    targets = df[target_col].values
    timestamps = df['DateTime'].values
    temp_predicted = df['Temperature_Predicted'].values

    X, y, ts, future_temps = [], [], [], []

    for i in range(lookback_steps, len(df)):
        sequence = features[i - lookback_steps:i].copy()

        if use_future_temp:
            future_temp = temp_predicted[i]
            future_temp_col = np.full((lookback_steps, 1), future_temp)
            sequence = np.hstack([sequence, future_temp_col])

        X.append(sequence)
        y.append(targets[i])
        ts.append(timestamps[i])
        future_temps.append(temp_predicted[i] if use_future_temp else None)

    return np.array(X), np.array(y), ts, future_temps

# Create training sequences
print(f"\nCreating LSTM sequences with {LOOKBACK_STEPS} timesteps lookback...")
X_train, y_train, timestamps_train, _ = create_lstm_sequences_with_future_temp(
    train_data, SEQUENCE_FEATURES, 'Consumption', LOOKBACK_STEPS, USE_FUTURE_TEMP
)

print(f"Training sequences shape: {X_train.shape}")

# 4 - NORMALIZE DATA
print("\n" + "="*70)
print("NORMALIZING DATA")
print("="*70)

n_features = X_train.shape[2]
n_timesteps = X_train.shape[1]

# Reshape for scaler
X_train_reshaped = X_train.reshape(-1, n_features)

scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
X_train_scaled = X_train_scaled.reshape(-1, n_timesteps, n_features)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

print(f"Normalized training data shape: {X_train_scaled.shape}")

# 5 - BUILD LSTM MODEL
print("\n" + "="*70)
print("BUILDING LSTM MODEL")
print("="*70)

def build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model

input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
model = build_lstm_model(input_shape, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)

print(f"\nModel Input Shape: {input_shape}")
print(f"LSTM Units: {LSTM_UNITS}")
model.summary()

# 6 - TRAIN THE MODEL
print("\n" + "="*70)
print("TRAINING LSTM MODEL")
print("="*70)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f}s")
print(f"Stopped at epoch: {len(history.history['loss'])}")

# 7 - FIND SIMILAR DAYS (for variance enhancement)
print("\n" + "="*70)
print("FINDING SIMILAR HISTORICAL DAYS")
print("="*70)

def find_similar_days(historical_data, target_date, n_similar=3):
    """Find days in historical data most similar to target date."""
    target_dow = pd.Timestamp(target_date).dayofweek
    target_is_weekend = 1 if target_dow >= 5 else 0

    # Get unique historical dates
    historical_dates = historical_data['Date'].unique()

    similarities = []
    for hist_date in historical_dates:
        hist_dow = pd.Timestamp(hist_date).dayofweek
        hist_is_weekend = 1 if hist_dow >= 5 else 0

        day_data = historical_data[historical_data['Date'] == hist_date]
        if len(day_data) < 200:
            continue

        hist_temp = day_data['Temperature'].mean()

        dow_diff = abs(target_dow - hist_dow)
        if dow_diff > 3:
            dow_diff = 7 - dow_diff

        weekend_match = 0 if target_is_weekend == hist_is_weekend else 2
        temp_diff = abs(test_data['Temperature'].mean() - hist_temp) if len(test_data) > 0 else 0

        score = dow_diff + weekend_match + temp_diff * 0.5

        similarities.append({
            'date': hist_date,
            'score': score,
            'dow': hist_dow,
            'temp': hist_temp,
            'consumption_mean': day_data['Consumption'].mean(),
            'consumption_std': day_data['Consumption'].std()
        })

    similarities.sort(key=lambda x: x['score'])
    return similarities[:n_similar]

similar_days = find_similar_days(train_data, target_date, SIMILAR_DAYS_COUNT)
print(f"\nTop {SIMILAR_DAYS_COUNT} similar days:")
for i, day in enumerate(similar_days):
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"  {i+1}. {day['date']} ({dow_names[day['dow']]}) - "
          f"Mean: {day['consumption_mean']:.0f} kW, Std: {day['consumption_std']:.0f} kW")

# Get patterns from similar days
similar_patterns = []
for day in similar_days:
    day_data = train_data[train_data['Date'] == day['date']].copy()
    day_data = day_data.sort_values('DateTime').reset_index(drop=True)
    if len(day_data) == 288:
        similar_patterns.append(day_data['Consumption'].values)

if similar_patterns:
    similar_patterns = np.array(similar_patterns)
    avg_pattern = np.mean(similar_patterns, axis=0)
    std_pattern = np.std(similar_patterns, axis=0)
else:
    avg_pattern = np.zeros(288)
    std_pattern = np.zeros(288)

# Get historical stats by time
hist_by_time = train_data.groupby(train_data['DateTime'].dt.time)['Consumption'].agg(['mean', 'std'])

# 8 - AUTOREGRESSIVE PREDICTION FOR TARGET DATE
print("\n" + "="*70)
print(f"AUTOREGRESSIVE PREDICTION FOR {TARGET_DATE}")
print(f"Using same approach as Forecast script (predictions feed back)")
print("="*70)

# Get the last LOOKBACK_STEPS rows before target date
rolling_buffer = train_data.tail(LOOKBACK_STEPS).copy().reset_index(drop=True)

if len(rolling_buffer) < LOOKBACK_STEPS:
    raise ValueError(f"Insufficient data: {len(rolling_buffer)}/{LOOKBACK_STEPS}")

# Generate timestamps for the target date
pred_start = pd.Timestamp(f"{TARGET_DATE} 00:00:00")
pred_end = pd.Timestamp(f"{TARGET_DATE} 23:55:00")
pred_timestamps = pd.date_range(start=pred_start, end=pred_end, freq='5min')

print(f"Generating {len(pred_timestamps)} predictions using autoregressive approach")

predictions_raw = []
predictions_enhanced = []
valid_timestamps = []

for i, pred_ts in enumerate(pred_timestamps):
    # Get temperature for this timestamp from test data
    temp_row = test_data[test_data['DateTime'] == pred_ts]
    if len(temp_row) > 0:
        future_temp = temp_row['Temperature_Predicted'].values[0]
        temp_actual = temp_row['Temperature'].values[0]
    else:
        future_temp = rolling_buffer['Temperature_Predicted'].iloc[-1]
        temp_actual = rolling_buffer['Temperature'].iloc[-1]

    # Build sequence from rolling buffer
    sequence_data = rolling_buffer[SEQUENCE_FEATURES].values.copy()

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
        time_key = pred_ts.time()
        if time_key in hist_by_time.index:
            hist_std = hist_by_time.loc[time_key, 'std']
            residual = np.random.normal(0, hist_std * BLEND_RATIO)
            pred_enhanced = pred_consumption + residual
        else:
            pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'pattern_match':
        if len(similar_patterns) > 0 and i < len(avg_pattern):
            pattern_deviation = avg_pattern[i] - avg_pattern.mean()
            pattern_std_at_time = std_pattern[i] if i < len(std_pattern) else std_pattern.mean()

            pred_enhanced = pred_consumption + pattern_deviation * BLEND_RATIO

            if pattern_std_at_time > 0:
                noise = np.random.normal(0, pattern_std_at_time * BLEND_RATIO * 0.5)
                pred_enhanced += noise
        else:
            pred_enhanced = pred_consumption

    elif VARIANCE_MODE == 'volatility_scale':
        time_key = pred_ts.time()
        if time_key in hist_by_time.index:
            hist_mean = hist_by_time.loc[time_key, 'mean']
            hist_std = hist_by_time.loc[time_key, 'std']

            if hist_std > 0:
                z_score = (pred_consumption - hist_mean) / (hist_std * 0.5)
                pred_enhanced = hist_mean + z_score * hist_std * (1 + BLEND_RATIO)
            else:
                pred_enhanced = pred_consumption
        else:
            pred_enhanced = pred_consumption
    else:
        pred_enhanced = pred_consumption

    pred_enhanced = max(pred_enhanced, 0)
    predictions_enhanced.append(pred_enhanced)
    valid_timestamps.append(pred_ts)

    # Update rolling buffer with enhanced prediction
    new_row = pd.DataFrame({
        'DateTime': [pred_ts],
        'Consumption': [pred_enhanced],
        'Temperature': [temp_actual],
        'Temperature_Predicted': [future_temp],
        'Hour_sin': [np.sin(2 * np.pi * pred_ts.hour / 24)],
        'Hour_cos': [np.cos(2 * np.pi * pred_ts.hour / 24)],
        'DayOfWeek_sin': [np.sin(2 * np.pi * pred_ts.dayofweek / 7)],
        'DayOfWeek_cos': [np.cos(2 * np.pi * pred_ts.dayofweek / 7)],
        'IsWeekend': [1 if pred_ts.dayofweek >= 5 else 0]
    })

    rolling_buffer = pd.concat([rolling_buffer.iloc[1:], new_row], ignore_index=True)

    if (i + 1) % 12 == 0:
        print(f"  {pred_ts.strftime('%H:%M')} - Raw: {pred_consumption:.1f} kW, Enhanced: {pred_enhanced:.1f} kW")

predictions_raw = np.array(predictions_raw)
predictions_enhanced = np.array(predictions_enhanced)

# Get actual consumption for comparison
y_actual = test_data.sort_values('DateTime')['Consumption'].values
timestamps_actual = test_data.sort_values('DateTime')['DateTime'].values

# 9 - CALCULATE METRICS
print("\n" + "="*70)
print("CALCULATING PERFORMANCE METRICS (AUTOREGRESSIVE)")
print("="*70)

# Align predictions with actual values
min_len = min(len(predictions_raw), len(y_actual))
pred_raw_aligned = predictions_raw[:min_len]
pred_enh_aligned = predictions_enhanced[:min_len]
actual_aligned = y_actual[:min_len]

# Raw LSTM metrics
mae_raw = mean_absolute_error(actual_aligned, pred_raw_aligned)
rmse_raw = np.sqrt(mean_squared_error(actual_aligned, pred_raw_aligned))
r2_raw = r2_score(actual_aligned, pred_raw_aligned)

# Enhanced metrics
mae_enh = mean_absolute_error(actual_aligned, pred_enh_aligned)
rmse_enh = np.sqrt(mean_squared_error(actual_aligned, pred_enh_aligned))
r2_enh = r2_score(actual_aligned, pred_enh_aligned)

# Training metrics
y_pred_train_scaled = model.predict(X_train_scaled, verbose=0)
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled).flatten()
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

print(f"\n{'Metric':<12} {'Train':>12} {'Raw LSTM':>12} {'Enhanced':>12}")
print("-" * 50)
print(f"{'R²':<12} {r2_train:>12.4f} {r2_raw:>12.4f} {r2_enh:>12.4f}")
print(f"{'MAE (kW)':<12} {mae_train:>12.2f} {mae_raw:>12.2f} {mae_enh:>12.2f}")
print(f"{'RMSE (kW)':<12} {rmse_train:>12.2f} {rmse_raw:>12.2f} {rmse_enh:>12.2f}")

# Variance comparison
print(f"\n--- Variance Comparison ---")
print(f"Actual std:     {actual_aligned.std():.1f} kW")
print(f"Raw LSTM std:   {pred_raw_aligned.std():.1f} kW ({pred_raw_aligned.std()/actual_aligned.std()*100:.0f}% of actual)")
print(f"Enhanced std:   {pred_enh_aligned.std():.1f} kW ({pred_enh_aligned.std()/actual_aligned.std()*100:.0f}% of actual)")

# 10 - CREATE RESULTS
print("\n" + "="*70)
print(f"AUTOREGRESSIVE PREDICTION RESULTS FOR {TARGET_DATE}")
print("="*70)

results_df = pd.DataFrame({
    'DateTime': valid_timestamps[:min_len],
    'Time': [ts.strftime('%H:%M') for ts in valid_timestamps[:min_len]],
    'Actual_kW': actual_aligned,
    'Raw_Prediction_kW': pred_raw_aligned,
    'Enhanced_Prediction_kW': pred_enh_aligned,
    'Raw_Error_kW': pred_raw_aligned - actual_aligned,
    'Enhanced_Error_kW': pred_enh_aligned - actual_aligned
})

results_filename = f'Cons_LSTM_Prediction_AR_{TARGET_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"Results saved to: {results_filename}")

# 11 - PLOT RESULTS
print("\nGenerating plots...")

fig, axes = plt.subplots(3, 1, figsize=(16, 16))

x_vals = np.arange(min_len)

# Plot 1: Actual vs Predicted
ax1 = axes[0]
ax1.plot(x_vals, actual_aligned, 'g-', label='Actual Consumption', linewidth=2)
ax1.plot(x_vals, pred_raw_aligned, 'b--', label='Raw LSTM (Autoregressive)', linewidth=1.5, alpha=0.7)
ax1.plot(x_vals, pred_enh_aligned, 'r-', label=f'Enhanced ({VARIANCE_MODE})', linewidth=2)

# Y-axis scale
y_min = min(min(actual_aligned), min(pred_enh_aligned), min(pred_raw_aligned)) * 0.9
y_max = max(max(actual_aligned), max(pred_enh_aligned), max(pred_raw_aligned)) * 1.05
ax1.set_ylim(y_min, y_max)

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Consumption (kW)', fontsize=12)
ax1.set_title(f'LSTM Autoregressive Prediction for {TARGET_DATE}\n'
             f'Raw R²: {r2_raw:.4f}, Enhanced R²: {r2_enh:.4f} | Mode: {VARIANCE_MODE}',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

tick_positions = np.arange(0, min_len, 24)
tick_labels = [valid_timestamps[i].strftime('%H:%M') for i in tick_positions if i < min_len]
ax1.set_xticks(tick_positions[:len(tick_labels)])
ax1.set_xticklabels(tick_labels, rotation=0, ha='center')

# Plot 2: Error comparison
ax2 = axes[1]
ax2.bar(x_vals - 0.2, results_df['Raw_Error_kW'].values, width=0.4, alpha=0.7,
        label=f'Raw LSTM Error (MAE: {mae_raw:.1f} kW)', color='blue')
ax2.bar(x_vals + 0.2, results_df['Enhanced_Error_kW'].values, width=0.4, alpha=0.7,
        label=f'Enhanced Error (MAE: {mae_enh:.1f} kW)', color='red')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_ylabel('Prediction Error (kW)', fontsize=12)
ax2.set_title('Prediction Error Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(tick_positions[:len(tick_labels)])
ax2.set_xticklabels(tick_labels, rotation=0, ha='center')

# Plot 3: Training history
ax3 = axes[2]
ax3.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax3.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss (MSE)', fontsize=12)
ax3.set_title('LSTM Training History', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f'Cons_LSTM_Prediction_AR_{TARGET_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plots saved to: {plot_filename}")
plt.show()

# 12 - FINAL SUMMARY
print("\n" + "#"*70)
print("### FINAL SUMMARY - AUTOREGRESSIVE LSTM PREDICTION")
print("#"*70)

print(f"\nTarget Date: {TARGET_DATE}")
print(f"Variance Enhancement Mode: {VARIANCE_MODE}")
print(f"Blend Ratio: {BLEND_RATIO}")

print(f"\n--- Model Performance (AUTOREGRESSIVE - Realistic) ---")
print(f"Raw LSTM:  R²={r2_raw:.4f}, MAE={mae_raw:.2f} kW, RMSE={rmse_raw:.2f} kW")
print(f"Enhanced:  R²={r2_enh:.4f}, MAE={mae_enh:.2f} kW, RMSE={rmse_enh:.2f} kW")

print(f"\n--- Variance Recovery ---")
print(f"Actual variance:   {actual_aligned.std():.1f} kW")
print(f"Raw LSTM recovery: {pred_raw_aligned.std()/actual_aligned.std()*100:.0f}%")
print(f"Enhanced recovery: {pred_enh_aligned.std()/actual_aligned.std()*100:.0f}%")

print(f"\n--- Daily Summary ---")
print(f"Total Actual:    {actual_aligned.sum() / 12:.2f} kWh")
print(f"Total Enhanced:  {pred_enh_aligned.sum() / 12:.2f} kWh")
daily_error = (pred_enh_aligned.sum() - actual_aligned.sum()) / 12
daily_error_pct = (pred_enh_aligned.sum() - actual_aligned.sum()) / actual_aligned.sum() * 100
print(f"Daily Error:     {daily_error:.2f} kWh ({daily_error_pct:.2f}%)")

print(f"\nComputation Time: {training_time:.2f}s")

# 13 - SAVE MODEL (if enabled)
if SAVE_MODEL:
    print("\n" + "="*70)
    print("SAVING LSTM MODEL FOR REAL FORECASTING")
    print("="*70)

    model_filename = 'Cons_LSTM_Model.keras'
    model.save(model_filename)
    print(f"Keras model saved to: {model_filename}")

    config_filename = 'Cons_LSTM_Config.npz'
    np.savez(config_filename,
             lookback_steps=np.array(LOOKBACK_STEPS),
             lstm_units=np.array(LSTM_UNITS),
             dropout_rate=np.array(DROPOUT_RATE),
             learning_rate=np.array(LEARNING_RATE),
             use_future_temp=np.array(USE_FUTURE_TEMP),
             feature_columns=np.array(FEATURE_COLUMNS),
             sequence_features=np.array(SEQUENCE_FEATURES),
             n_features=np.array(len(FEATURE_COLUMNS)),
             scaler_X_min=scaler_X.data_min_,
             scaler_X_max=scaler_X.data_max_,
             scaler_X_scale=scaler_X.scale_,
             scaler_X_data_range=scaler_X.data_range_,
             scaler_y_min=scaler_y.data_min_,
             scaler_y_max=scaler_y.data_max_,
             scaler_y_scale=scaler_y.scale_,
             scaler_y_data_range=scaler_y.data_range_,
             training_r2=np.array(r2_train),
             training_mae=np.array(mae_train),
             training_rmse=np.array(rmse_train),
             epochs_trained=np.array(len(history.history['loss']))
    )

    print(f"Configuration saved to: {config_filename}")
    print("="*70)

print("\n" + "#"*70)
print("### AUTOREGRESSIVE LSTM PREDICTION COMPLETE!")
print("#"*70)
