"""
LSTM-based Consumption Forecasting
==================================
This script uses a Long Short-Term Memory (LSTM) neural network for time series forecasting.

KEY DIFFERENCES FROM FEEDFORWARD DNN:
-------------------------------------
1. LSTM processes data SEQUENTIALLY - it "remembers" patterns over time
2. Input shape is 3D: (samples, timesteps, features) instead of 2D: (samples, features)
3. LSTM has internal "memory cells" that learn what to remember/forget from past data
4. Better suited for time series because it captures temporal dependencies

LSTM ARCHITECTURE EXPLAINED:
----------------------------
- Input Gate: Controls what new information to store in memory
- Forget Gate: Controls what information to discard from memory
- Output Gate: Controls what information to output based on memory
- Cell State: The "memory" that flows through time, carrying relevant information

Reference: https://www.tensorflow.org/tutorials/structured_data/time_series
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
LEARNING_RATE = 0.001                # Learning rate (lower for LSTM, typically 0.001)
EPOCHS = 100                         # Maximum training epochs
BATCH_SIZE = 32                      # Number of samples per gradient update
LOOKBACK_STEPS = 288                 # 288 x 5min = 24 hours of past data (reduced for LSTM)
LSTM_UNITS = 64                      # Number of LSTM units (memory cells)
DROPOUT_RATE = 0.2                   # Dropout for regularization (prevents overfitting)
PATIENCE = 10                        # Early stopping patience
SAVE_MODEL = True                    # Save model for real forecasting
# ========================================================================
# TARGET DATE TO PREDICT (excluded from training)
TARGET_DATE = '2026-01-27'
# ========================================================================

# 1 - LOAD AND PREPROCESS DATA
print("="*70)
print("LOADING AND PREPROCESSING DATA FOR LSTM CONSUMPTION FORECASTING")
print(f"Target prediction date: {TARGET_DATE}")
print(f"Resolution: 5 minutes")
print("="*70)

# Load data from CSV (skip first 3 header rows)
data = pd.read_csv('../Data_January.csv', skiprows=3, header=None, encoding='latin-1')
print(f"Loaded data shape: {data.shape}")

# Assign column names based on the file structure
data.columns = ['DateTime_str', 'Date', 'DayTime', 'Forecast_Prod', 'Forecast_Load',
                'Consumption', 'Production', 'Level_Bidmi', 'Level_Haselholz',
                'Temperature', 'Irradiance', 'Rain', 'SDR_Mode', 'Forecast_Mode',
                'Transfer_Mode', 'Waterlevel_Mode', 'Temp_Forecast']

print(f"Columns: {list(data.columns)}")

# Parse DateTime
data['DateTime'] = pd.to_datetime(data['DateTime_str'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
data = data.dropna(subset=['DateTime'])
data = data.sort_values('DateTime').reset_index(drop=True)

# Handle missing Temp_Forecast values (use actual temperature where forecast is missing)
data['Temp_Forecast'] = pd.to_numeric(data['Temp_Forecast'], errors='coerce')
data['Temp_Forecast'] = data['Temp_Forecast'].fillna(data['Temperature'])

print(f"Data shape after parsing: {data.shape}")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

# Use 5-minute data directly (no aggregation needed)
print("\nUsing native 5-minute resolution data...")

# Rename columns for clarity
data.rename(columns={'Temp_Forecast': 'Temperature_Predicted'}, inplace=True)

# Assign to data_5min
data_5min = data.copy()

print(f"5-minute data shape: {data_5min.shape}")

# Extract time features
data_5min['Hour'] = data_5min['DateTime'].dt.hour
data_5min['Minute'] = data_5min['DateTime'].dt.minute
data_5min['DayOfWeek'] = data_5min['DateTime'].dt.dayofweek
data_5min['Month'] = data_5min['DateTime'].dt.month
data_5min['DayOfYear'] = data_5min['DateTime'].dt.dayofyear
data_5min['IsWeekend'] = (data_5min['DayOfWeek'] >= 5).astype(int)
data_5min['Date'] = data_5min['DateTime'].dt.date

# Create cyclical time features (helps LSTM understand time patterns)
# Sin/Cos encoding ensures 23:55 is close to 00:00 (circular time)
data_5min['Hour_sin'] = np.sin(2 * np.pi * data_5min['Hour'] / 24)
data_5min['Hour_cos'] = np.cos(2 * np.pi * data_5min['Hour'] / 24)
data_5min['DayOfWeek_sin'] = np.sin(2 * np.pi * data_5min['DayOfWeek'] / 7)
data_5min['DayOfWeek_cos'] = np.cos(2 * np.pi * data_5min['DayOfWeek'] / 7)

print(f"\n5-minute data preview:")
print(data_5min.head(10))

# 2 - SPLIT DATA: EXCLUDE TARGET DATE FROM TRAINING
print("\n" + "="*70)
print(f"SPLITTING DATA - EXCLUDING {TARGET_DATE} FOR PREDICTION")
print("="*70)

target_date = pd.to_datetime(TARGET_DATE).date()

# Training data: all days BEFORE target date (exclude target date and future data)
# This prevents data leakage - we can't use future data to predict the past
train_data = data_5min[data_5min['Date'] < target_date].copy()

# Test data: target date only
test_data = data_5min[data_5min['Date'] == target_date].copy()

print(f"Training data: {len(train_data)} intervals (5-min)")
print(f"Test data ({TARGET_DATE}): {len(test_data)} intervals (5-min)")

# We need enough days before for lookback features
days_needed = int(np.ceil(LOOKBACK_STEPS / 288)) + 1
lookback_start = pd.to_datetime(TARGET_DATE) - pd.Timedelta(days=days_needed)
lookback_data = data_5min[(data_5min['DateTime'] >= lookback_start) &
                          (data_5min['Date'] < target_date)].copy()
print(f"Lookback data: {len(lookback_data)} intervals (5-min) from {days_needed} days before")

# 3 - PREPARE FEATURES FOR LSTM
print("\n" + "="*70)
print("PREPARING FEATURES FOR LSTM")
print("="*70)

# Define which features to use for the LSTM SEQUENCE
# LSTM will process these as a SEQUENCE over time
SEQUENCE_FEATURES = [
    'Consumption',           # Main feature: past consumption values
    'Temperature',           # Actual temperature
    'Temperature_Predicted', # Forecasted temperature
    'Hour_sin',             # Cyclical hour encoding (sin component)
    'Hour_cos',             # Cyclical hour encoding (cos component)
    'DayOfWeek_sin',        # Cyclical day encoding (sin component)
    'DayOfWeek_cos',        # Cyclical day encoding (cos component)
    'IsWeekend'             # Binary: is it weekend?
]

# We also add the FUTURE temperature forecast as an extra feature
# This is appended to every timestep in the sequence (same value repeated)
# This allows the LSTM to "know" what temperature is expected at prediction time
USE_FUTURE_TEMP = True

print(f"Sequence features: {SEQUENCE_FEATURES}")
print(f"Number of sequence features per timestep: {len(SEQUENCE_FEATURES)}")
print(f"Using future temperature forecast: {USE_FUTURE_TEMP}")

def create_lstm_sequences_with_future_temp(df, sequence_cols, target_col, lookback_steps, use_future_temp=True):
    """
    Create sequences for LSTM training with optional future temperature forecast.

    LSTM INPUT SHAPE EXPLANATION:
    -----------------------------
    Unlike feedforward networks that see all features at once (flat vector),
    LSTM processes data as SEQUENCES:

    Input shape: (num_samples, timesteps, num_features)
    - num_samples: How many sequences we have
    - timesteps: How many time steps in each sequence (LOOKBACK_STEPS)
    - num_features: How many features at each time step

    WITH FUTURE TEMPERATURE:
    -----------------------
    We add an extra feature column that contains the temperature forecast
    for the TARGET timestep (the one we're predicting). This value is the
    same across all timesteps in the sequence, allowing the LSTM to use
    this future information when making its prediction.

    Visual representation of one sample (with future temp):

    Time    | Consumption | Temperature | ... | Future_Temp |
    --------|-------------|-------------|-----|-------------|
    t-288   |    2100     |     5.2     | ... |     7.5     |  <- Same future temp
    t-287   |    2050     |     5.1     | ... |     7.5     |  <- for all rows
    ...     |    ...      |    ...      | ... |     7.5     |
    t-1     |    2200     |     6.0     | ... |     7.5     |

    Target: Consumption at time t (where temp forecast is 7.5)
    """
    features = df[sequence_cols].values
    targets = df[target_col].values
    timestamps = df['DateTime'].values
    temp_predicted = df['Temperature_Predicted'].values

    X, y, ts, future_temps = [], [], [], []

    # Slide a window of size 'lookback_steps' through the data
    for i in range(lookback_steps, len(df)):
        # Extract sequence: from (i - lookback_steps) to i
        sequence = features[i - lookback_steps:i].copy()

        if use_future_temp:
            # Get the temperature forecast for the TARGET timestep
            future_temp = temp_predicted[i]

            # Add future temp as an extra column (same value for all timesteps)
            future_temp_col = np.full((lookback_steps, 1), future_temp)
            sequence = np.hstack([sequence, future_temp_col])

        X.append(sequence)
        y.append(targets[i])
        ts.append(timestamps[i])
        future_temps.append(temp_predicted[i] if use_future_temp else None)

    return np.array(X), np.array(y), ts, future_temps

# For saving - include future temp info
FEATURE_COLUMNS = SEQUENCE_FEATURES + (['Future_Temp_Forecast'] if USE_FUTURE_TEMP else [])

# Create training sequences
print(f"\nCreating LSTM sequences with {LOOKBACK_STEPS} timesteps lookback...")
X_train, y_train, timestamps_train, _ = create_lstm_sequences_with_future_temp(
    train_data, SEQUENCE_FEATURES, 'Consumption', LOOKBACK_STEPS, USE_FUTURE_TEMP
)

print(f"Training sequences shape: {X_train.shape}")
print(f"  - {X_train.shape[0]} samples")
print(f"  - {X_train.shape[1]} timesteps per sample")
print(f"  - {X_train.shape[2]} features per timestep")
print(f"Training targets shape: {y_train.shape}")

# 4 - CREATE SEQUENCES FOR TARGET DATE PREDICTION
print("\n" + "="*70)
print(f"CREATING SEQUENCES FOR {TARGET_DATE} PREDICTION")
print("="*70)

# Combine lookback data with test data for creating test sequences
combined_data = pd.concat([lookback_data, test_data]).reset_index(drop=True)
print(f"Combined data for prediction: {len(combined_data)} intervals (5-min)")

# Create test sequences
X_test, y_test, timestamps_test, _ = create_lstm_sequences_with_future_temp(
    combined_data, SEQUENCE_FEATURES, 'Consumption', LOOKBACK_STEPS, USE_FUTURE_TEMP
)

# Filter to only include predictions for the target date
target_date_mask = [pd.Timestamp(ts).date() == target_date for ts in timestamps_test]
X_test = X_test[target_date_mask]
y_test = y_test[target_date_mask]
timestamps_test = [ts for ts, mask in zip(timestamps_test, target_date_mask) if mask]

print(f"Test sequences shape: {X_test.shape}")
print(f"Test targets shape: {y_test.shape}")
print(f"5-min intervals to predict: {len(timestamps_test)}")

# 5 - NORMALIZE DATA
print("\n" + "="*70)
print("NORMALIZING DATA")
print("="*70)

"""
NORMALIZATION FOR LSTM:
-----------------------
We normalize each feature to [0, 1] range because:
1. LSTM uses sigmoid/tanh activations that work best with small values
2. Features on different scales (kW vs temperature) would dominate training
3. Helps gradient descent converge faster

IMPORTANT: We fit the scaler ONLY on training data to avoid data leakage!
"""

# Normalize features (fit on training data only)
n_features = X_train.shape[2]
n_timesteps = X_train.shape[1]

# Reshape for scaler: (samples * timesteps, features)
X_train_reshaped = X_train.reshape(-1, n_features)
X_test_reshaped = X_test.reshape(-1, n_features)

scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
X_test_scaled = scaler_X.transform(X_test_reshaped)

# Reshape back to 3D: (samples, timesteps, features)
X_train_scaled = X_train_scaled.reshape(-1, n_timesteps, n_features)
X_test_scaled = X_test_scaled.reshape(-1, n_timesteps, n_features)

# Normalize target
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"Normalized training data shape: {X_train_scaled.shape}")
print(f"Normalized test data shape: {X_test_scaled.shape}")

# 6 - BUILD LSTM MODEL
print("\n" + "="*70)
print("BUILDING LSTM MODEL")
print("="*70)

"""
LSTM MODEL ARCHITECTURE:
------------------------

Sequential model with:
1. LSTM Layer: The core recurrent layer with memory cells
2. Dropout Layer: Randomly drops neurons during training (prevents overfitting)
3. Dense Layer: Final output layer for prediction

LSTM Parameters explained:
- units: Number of LSTM cells (memory capacity)
- return_sequences:
    - True: Output at every timestep (for stacking LSTMs)
    - False: Output only at the last timestep (for final prediction)
- input_shape: (timesteps, features) - defines what each sample looks like

Visual of LSTM processing a sequence:

    Input sequence:     [x1] -> [x2] -> [x3] -> ... -> [x288]
                         |       |       |              |
    LSTM cells:         [h1] -> [h2] -> [h3] -> ... -> [h288]
                                                        |
    Output (return_sequences=False):              [prediction]

Where each [xi] is a vector of 8 features, and each [hi] is the hidden state
"""

def build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate):
    """
    Build and compile an LSTM model for time series forecasting.

    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    lstm_units : int
        Number of LSTM units (memory cells)
    dropout_rate : float
        Fraction of units to drop (0-1)
    learning_rate : float
        Learning rate for Adam optimizer

    Returns:
    --------
    model : keras.Model
        Compiled LSTM model ready for training
    """
    model = Sequential([
        # First LSTM layer
        # - units: number of memory cells
        # - return_sequences=True: needed when stacking LSTM layers
        # - input_shape: (timesteps, features)
        LSTM(units=lstm_units,
             return_sequences=True,  # Output sequence for next LSTM layer
             input_shape=input_shape),

        # Dropout for regularization (prevents overfitting)
        # Randomly sets 20% of inputs to 0 during training
        Dropout(dropout_rate),

        # Second LSTM layer
        # return_sequences=False: only output the final hidden state
        LSTM(units=lstm_units // 2,  # Fewer units in second layer
             return_sequences=False),

        Dropout(dropout_rate),

        # Dense layer to map LSTM output to prediction
        Dense(units=32, activation='relu'),

        # Output layer: single neuron for consumption prediction
        Dense(units=1)
    ])

    # Compile model with Adam optimizer and MSE loss
    # Adam: Adaptive learning rate optimizer (good default choice)
    # MSE: Mean Squared Error (standard for regression)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']  # Also track Mean Absolute Error
    )

    return model

# Build the model
input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
model = build_lstm_model(input_shape, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)

# Print model summary
print(f"\nModel Input Shape: {input_shape}")
print(f"LSTM Units: {LSTM_UNITS}")
print(f"Dropout Rate: {DROPOUT_RATE}")
print(f"Learning Rate: {LEARNING_RATE}")
print("\nModel Architecture:")
model.summary()

# 7 - TRAIN THE MODEL
print("\n" + "="*70)
print("TRAINING LSTM MODEL")
print(f"Epochs: {EPOCHS} (with early stopping, patience={PATIENCE})")
print(f"Batch Size: {BATCH_SIZE}")
print("="*70)

"""
TRAINING PROCESS:
-----------------
- Epochs: Number of complete passes through training data
- Batch Size: Number of samples processed before updating weights
- Early Stopping: Stop training if validation loss doesn't improve
  (prevents overfitting and saves time)
- Validation Split: Use 20% of training data to monitor overfitting
"""

# Early stopping callback
# Monitors validation loss and stops if no improvement for 'patience' epochs
early_stopping = EarlyStopping(
    monitor='val_loss',      # Watch validation loss
    patience=PATIENCE,       # Stop after 10 epochs without improvement
    restore_best_weights=True,  # Keep the best model, not the last one
    verbose=1
)

start_time = time.time()

# Train the model
history = model.fit(
    X_train_scaled,           # Training sequences
    y_train_scaled,           # Training targets
    epochs=EPOCHS,            # Maximum epochs
    batch_size=BATCH_SIZE,    # Samples per gradient update
    validation_split=0.2,     # Use 20% for validation
    callbacks=[early_stopping],  # Early stopping
    verbose=1                 # Show progress
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f}s ({training_time/60:.2f} min)")
print(f"Stopped at epoch: {len(history.history['loss'])}")

# 8 - MAKE PREDICTIONS
print("\n" + "="*70)
print(f"PREDICTING CONSUMPTION FOR {TARGET_DATE}")
print("="*70)

# Predict on training data (for comparison)
y_pred_train_scaled = model.predict(X_train_scaled, verbose=0)
y_pred_train_inv = scaler_y.inverse_transform(y_pred_train_scaled).flatten()
y_actual_train_inv = y_train

# Predict on test data (target date)
y_pred_test_scaled = model.predict(X_test_scaled, verbose=0)
y_pred_test_inv = scaler_y.inverse_transform(y_pred_test_scaled).flatten()
y_actual_test_inv = y_test

# 9 - CALCULATE METRICS
print("\n" + "="*70)
print("CALCULATING PERFORMANCE METRICS")
print("="*70)

# Training metrics
mae_train = mean_absolute_error(y_actual_train_inv, y_pred_train_inv)
mse_train = mean_squared_error(y_actual_train_inv, y_pred_train_inv)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_actual_train_inv, y_pred_train_inv)

# Test metrics
mae_test = mean_absolute_error(y_actual_test_inv, y_pred_test_inv)
mse_test = mean_squared_error(y_actual_test_inv, y_pred_test_inv)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_actual_test_inv, y_pred_test_inv)

# 10 - PRINT RESULTS
print("\n" + "="*70)
print("RESULTS - LSTM CONSUMPTION FORECASTING (5-min resolution)")
print("="*70)
print(f"{'Metric':<12} {'Train':>14} {f'{TARGET_DATE}':>14} {'Diff':>14}")
print("-" * 70)
print(f"{'R²':<12} {r2_train:>14.6f} {r2_test:>14.6f} {r2_test - r2_train:>14.6f}")
print(f"{'MAE (kW)':<12} {mae_train:>14.2f} {mae_test:>14.2f} {mae_test - mae_train:>14.2f}")
print(f"{'MSE':<12} {mse_train:>14.2f} {mse_test:>14.2f} {mse_test - mse_train:>14.2f}")
print(f"{'RMSE (kW)':<12} {rmse_train:>14.2f} {rmse_test:>14.2f} {rmse_test - rmse_train:>14.2f}")
print("-" * 70)

# Diagnostic
if r2_train > 0.9 and r2_test < 0.7:
    diagnostic = "OVERFITTING: High Train R², Low Test R²"
elif r2_train < 0.5 and r2_test < 0.5:
    diagnostic = "UNDERFITTING: Poor performance on Train and Test"
elif abs(r2_train - r2_test) < 0.15:
    diagnostic = "GOOD FIT: Similar performance on Train/Test"
else:
    diagnostic = "POSSIBLE OVERFITTING: Notable gap between Train and Test"

print(f"Diagnostic: {diagnostic}")
print(f"Computation Time: {training_time:.2f}s ({training_time/60:.2f} min)")
print("="*70)

# 11 - CREATE RESULTS DATAFRAME
print("\n" + "="*70)
print(f"5-MINUTE CONSUMPTION FORECAST FOR {TARGET_DATE}")
print("="*70)

results_df = pd.DataFrame({
    'DateTime': timestamps_test,
    'Time': [pd.Timestamp(ts).strftime('%H:%M') for ts in timestamps_test],
    'Actual_Consumption_kW': y_actual_test_inv,
    'Predicted_Consumption_kW': y_pred_test_inv,
    'Error_kW': y_pred_test_inv - y_actual_test_inv,
    'Error_%': ((y_pred_test_inv - y_actual_test_inv) / y_actual_test_inv * 100)
})

print(results_df.to_string(index=False))

# Save results to CSV
results_filename = f'Cons_LSTM_Prediction_{TARGET_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nResults saved to: {results_filename}")

# 12 - PLOT RESULTS
print("\nGenerating plots...")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Plot 1: Actual vs Predicted
ax1 = axes[0]
x_vals = np.arange(len(timestamps_test))
ax1.plot(x_vals, y_actual_test_inv, 'b-', label='Actual Consumption', linewidth=1.5)
ax1.plot(x_vals, y_pred_test_inv, 'r--', label='Predicted Consumption', linewidth=1.5)
ax1.fill_between(x_vals, y_actual_test_inv, y_pred_test_inv, alpha=0.3, color='gray')

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Consumption (kW)', fontsize=12)
ax1.set_title(f'LSTM Consumption Forecast for {TARGET_DATE} (5-min resolution)\nR²: {r2_test:.4f}, RMSE: {rmse_test:.2f} kW, MAE: {mae_test:.2f} kW',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# X-axis labels every hour
tick_positions = np.arange(0, len(timestamps_test), 12)
tick_labels = [pd.Timestamp(timestamps_test[i]).strftime('%H:%M')
               for i in tick_positions if i < len(timestamps_test)]
ax1.set_xticks(tick_positions[:len(tick_labels)])
ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

# Plot 2: Prediction Error
ax2 = axes[1]
colors = ['green' if e < 0 else 'red' for e in results_df['Error_kW']]
ax2.bar(x_vals, results_df['Error_kW'], color=colors, edgecolor='none', alpha=0.7, width=1.0)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axhline(y=mae_test, color='red', linestyle='--', linewidth=1, label=f'MAE: {mae_test:.2f} kW')
ax2.axhline(y=-mae_test, color='red', linestyle='--', linewidth=1)

ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_ylabel('Prediction Error (kW)', fontsize=12)
ax2.set_title('5-Minute Prediction Error (Predicted - Actual)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(tick_positions[:len(tick_labels)])
ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

# Plot 3: Training History (Loss over epochs)
ax3 = axes[2]
ax3.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax3.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss (MSE)', fontsize=12)
ax3.set_title('LSTM Training History', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f'Cons_LSTM_Prediction_{TARGET_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plots saved to: {plot_filename}")
plt.show()

# 13 - FINAL SUMMARY
print("\n" + "#"*70)
print("### FINAL SUMMARY - LSTM CONSUMPTION FORECASTING (5-min resolution)")
print("#"*70)

print(f"\nTarget Date: {TARGET_DATE}")
print(f"Resolution: 5 minutes")
print(f"Training Period: All data except {TARGET_DATE}")
print(f"Training Samples: {len(y_train)}")
print(f"Prediction Intervals: {len(y_test)}")

print(f"\n--- LSTM Model Configuration ---")
print(f"Lookback Steps: {LOOKBACK_STEPS} ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
print(f"Features per timestep: {len(FEATURE_COLUMNS)}")
print(f"LSTM Units: {LSTM_UNITS}")
print(f"Dropout Rate: {DROPOUT_RATE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs trained: {len(history.history['loss'])}")

print(f"\n--- Model Performance on {TARGET_DATE} ---")
print(f"R²:   {r2_test:.4f}")
print(f"MAE:  {mae_test:.2f} kW")
print(f"RMSE: {rmse_test:.2f} kW")
print(f"MAPE: {np.mean(np.abs(results_df['Error_%'])):.2f}%")

print(f"\n--- Daily Summary ---")
print(f"Total Actual Consumption:    {y_actual_test_inv.sum() / 12:.2f} kWh")
print(f"Total Predicted Consumption: {y_pred_test_inv.sum() / 12:.2f} kWh")
daily_error = (y_pred_test_inv.sum() - y_actual_test_inv.sum()) / 12
daily_error_pct = (y_pred_test_inv.sum() - y_actual_test_inv.sum()) / y_actual_test_inv.sum() * 100
print(f"Daily Error:                 {daily_error:.2f} kWh ({daily_error_pct:.2f}%)")

print(f"\nComputation Time: {training_time:.2f}s ({training_time/60:.2f} min)")

print("\n" + "#"*70)
print("### LSTM FORECASTING COMPLETE!")
print("#"*70)

# 14 - SAVE MODEL (if enabled)
if SAVE_MODEL:
    print("\n" + "="*70)
    print("SAVING LSTM MODEL FOR REAL FORECASTING")
    print("="*70)

    # Save Keras model
    model_filename = 'Cons_LSTM_Model.keras'
    model.save(model_filename)
    print(f"Keras model saved to: {model_filename}")

    # Save scalers and configuration to .npz file
    config_filename = 'Cons_LSTM_Config.npz'
    np.savez(config_filename,
             # Model configuration
             lookback_steps=np.array(LOOKBACK_STEPS),
             lstm_units=np.array(LSTM_UNITS),
             dropout_rate=np.array(DROPOUT_RATE),
             learning_rate=np.array(LEARNING_RATE),
             use_future_temp=np.array(USE_FUTURE_TEMP),
             # Feature columns (save as string)
             feature_columns=np.array(FEATURE_COLUMNS),
             sequence_features=np.array(SEQUENCE_FEATURES),
             n_features=np.array(len(FEATURE_COLUMNS)),
             # Scaler parameters for X (MinMaxScaler)
             scaler_X_min=scaler_X.data_min_,
             scaler_X_max=scaler_X.data_max_,
             scaler_X_scale=scaler_X.scale_,
             scaler_X_data_range=scaler_X.data_range_,
             # Scaler parameters for y (MinMaxScaler)
             scaler_y_min=scaler_y.data_min_,
             scaler_y_max=scaler_y.data_max_,
             scaler_y_scale=scaler_y.scale_,
             scaler_y_data_range=scaler_y.data_range_,
             # Training info
             training_r2=np.array(r2_train),
             training_mae=np.array(mae_train),
             training_rmse=np.array(rmse_train),
             epochs_trained=np.array(len(history.history['loss']))
    )

    print(f"Configuration saved to: {config_filename}")
    print(f"  - Lookback steps: {LOOKBACK_STEPS}")
    print(f"  - LSTM Units: {LSTM_UNITS}")
    print(f"  - Features: {FEATURE_COLUMNS}")
    print(f"  - Training R²: {r2_train:.4f}")
    print("="*70)

"""
TIPS FOR IMPROVING LSTM PERFORMANCE:
------------------------------------
1. Increase LOOKBACK_STEPS: More history = more context (but slower training)
2. Add more LSTM layers: Deeper networks can learn more complex patterns
3. Increase LSTM_UNITS: More memory cells = more capacity
4. Try Bidirectional LSTM: Processes sequence forwards AND backwards
5. Add more features: Weather forecasts, holidays, special events
6. Tune DROPOUT_RATE: Higher = less overfitting, but may underfit
7. Try different optimizers: SGD, RMSprop, AdaGrad
8. Use learning rate scheduling: Decrease LR as training progresses

COMMON ISSUES:
--------------
- Overfitting: Increase dropout, reduce model complexity, add more data
- Underfitting: Increase model complexity, train longer, add features
- Slow training: Reduce LOOKBACK_STEPS, use GPU, reduce batch size
"""
