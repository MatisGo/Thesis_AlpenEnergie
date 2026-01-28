import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from dnn_app_utils_v3 import *

# ========================================================================
# HYPERPARAMETERS - EASY TO CHANGE
# ========================================================================
LEARNING_RATE = 0.6                                   # Learning rate for gradient descent
NUM_ITERATIONS = 3000                                 # Number of iterations for training
LOOKBACK_STEPS = 576                                  # 288 x 5min = 24 hours of past data
PRINT_COST = False                                    # Print cost during training
# ========================================================================
# TARGET DATE TO PREDICT (excluded from training)
TARGET_DATE = '2026-01-18'
# ========================================================================

# 1 - LOAD AND PREPROCESS DATA
print("="*70)
print("LOADING AND PREPROCESSING DATA FOR CONSUMPTION FORECASTING")
print(f"Target prediction date: {TARGET_DATE}")
print(f"Resolution: 5 minutes")
print("="*70)

# Load data from CSV (skip first 3 header rows)
data = pd.read_csv('Data_January.csv', skiprows=3, header=None, encoding='latin-1')
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

print(f"\n5-minute data preview:")
print(data_5min.head(10))

# 2 - SPLIT DATA: EXCLUDE TARGET DATE AND FUTURE FROM TRAINING
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
# LOOKBACK_STEPS intervals at 5-min resolution = LOOKBACK_STEPS/288 days
days_needed = int(np.ceil(LOOKBACK_STEPS / 288)) + 1  # +1 for safety margin
lookback_start = pd.to_datetime(TARGET_DATE) - pd.Timedelta(days=days_needed)
lookback_data = data_5min[(data_5min['DateTime'] >= lookback_start) &
                          (data_5min['Date'] < target_date)].copy()
print(f"Lookback data: {len(lookback_data)} intervals (5-min) from {days_needed} days before")

# 3 - CREATE FEATURES FOR TRAINING
print("\n" + "="*70)
print("CREATING FEATURES FOR TRAINING")
print("="*70)

def create_forecast_features(df, lookback_steps=288, forecast_step=1):
    """
    Create features for forecasting consumption at a specific step ahead.
    lookback_steps=288 means 24 hours of 5-min data
    """
    features = []
    targets = []
    timestamps = []

    for i in range(lookback_steps, len(df) - forecast_step):
        feature_row = []

        # Past consumption (last 24 hours = 288 x 5min)
        past_consumption = df['Consumption'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_consumption)

        # Past temperature (last 24 hours)
        past_temperature = df['Temperature'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_temperature)

        # Past predicted temperature (last 24 hours)
        past_temp_pred = df['Temperature_Predicted'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_temp_pred)

        # Future predicted temperature (for the target step)
        future_temp_pred = df['Temperature_Predicted'].iloc[i + forecast_step]
        feature_row.append(future_temp_pred)

        # Time features for the target step
        target_datetime = df['DateTime'].iloc[i + forecast_step]
        feature_row.append(target_datetime.hour)
        feature_row.append(target_datetime.minute)
        feature_row.append(target_datetime.dayofweek)
        feature_row.append(target_datetime.month)
        feature_row.append(1 if target_datetime.dayofweek >= 5 else 0)

        features.append(feature_row)
        targets.append(df['Consumption'].iloc[i + forecast_step])
        timestamps.append(target_datetime)

    return np.array(features), np.array(targets), timestamps

# Create training features
print(f"Creating training features with {LOOKBACK_STEPS} steps lookback (24 hours)...")
X_train, y_train, timestamps_train = create_forecast_features(train_data, lookback_steps=LOOKBACK_STEPS, forecast_step=1)

# Define feature names
feature_names = []
for i in range(LOOKBACK_STEPS):
    feature_names.append(f'Consumption_t-{LOOKBACK_STEPS-i}')
for i in range(LOOKBACK_STEPS):
    feature_names.append(f'Temperature_t-{LOOKBACK_STEPS-i}')
for i in range(LOOKBACK_STEPS):
    feature_names.append(f'TempPred_t-{LOOKBACK_STEPS-i}')
feature_names.extend(['TempPred_future', 'Hour', 'Minute', 'DayOfWeek', 'Month', 'IsWeekend'])

print(f"Total features: {len(feature_names)}")
print(f"Training feature matrix shape: {X_train.shape}")
print(f"Training target vector shape: {y_train.shape}")

# 4 - CREATE FEATURES FOR JANUARY 19TH PREDICTION
print("\n" + "="*70)
print(f"CREATING FEATURES FOR {TARGET_DATE} PREDICTION")
print("="*70)

# Combine lookback data (Jan 18) with test data (Jan 19) for creating features
combined_data = pd.concat([lookback_data, test_data]).reset_index(drop=True)
print(f"Combined data for prediction: {len(combined_data)} intervals (5-min)")

# Create test features using the combined data
X_test_list = []
y_test_list = []
timestamps_test = []

# For each 15-min interval on January 19th, we need the previous 24 hours of data
for i in range(LOOKBACK_STEPS, len(combined_data)):
    feature_row = []

    # Past consumption (last 24 hours)
    past_consumption = combined_data['Consumption'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_consumption)

    # Past temperature (last 24 hours)
    past_temperature = combined_data['Temperature'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_temperature)

    # Past predicted temperature (last 24 hours)
    past_temp_pred = combined_data['Temperature_Predicted'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_temp_pred)

    # Future predicted temperature (using the forecast for the target step)
    future_temp_pred = combined_data['Temperature_Predicted'].iloc[i]
    feature_row.append(future_temp_pred)

    # Time features for the target step
    target_datetime = combined_data['DateTime'].iloc[i]
    feature_row.append(target_datetime.hour)
    feature_row.append(target_datetime.minute)
    feature_row.append(target_datetime.dayofweek)
    feature_row.append(target_datetime.month)
    feature_row.append(1 if target_datetime.dayofweek >= 5 else 0)

    X_test_list.append(feature_row)
    y_test_list.append(combined_data['Consumption'].iloc[i])
    timestamps_test.append(target_datetime)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

print(f"Test feature matrix shape: {X_test.shape}")
print(f"Test target vector shape: {y_test.shape}")
print(f"5-min intervals to predict: {len(timestamps_test)}")

# 5 - NORMALIZE AND PREPARE FOR NEURAL NETWORK
print("\n" + "="*70)
print("NORMALIZING DATA")
print("="*70)

# Update network architecture
n_features = X_train.shape[1]
LAYERS_DIMS = [n_features, 128, 64, 32, 16, 1]
print(f"Network architecture: {LAYERS_DIMS}")

# Normalize features
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Normalize target
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Reshape for neural network (features, samples)
y_train_nn = y_train_scaled.reshape(1, -1)
y_test_nn = y_test_scaled.reshape(1, -1)
X_train_nn = X_train_scaled.T
X_test_nn = X_test_scaled.T

print(f"Training samples: {X_train_nn.shape[1]}")
print(f"Test samples (Jan 19): {X_test_nn.shape[1]}")

# 6 - TRAIN THE MODEL
print("\n" + "="*70)
print("TRAINING MODEL FOR CONSUMPTION FORECASTING")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Iterations: {NUM_ITERATIONS}")
print(f"Architecture: {LAYERS_DIMS}")
print("="*70)

def L_layer_model(X, Y, layers_dims, learning_rate=0.5, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    """
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

start_time = time.time()

print(f"Training model with {NUM_ITERATIONS} iterations...")
parameters, costs = L_layer_model(X_train_nn, y_train_nn, LAYERS_DIMS,
                                  learning_rate=LEARNING_RATE,
                                  num_iterations=NUM_ITERATIONS,
                                  print_cost=PRINT_COST)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")

# 7 - MAKE PREDICTIONS FOR JANUARY 19TH
print("\n" + "="*70)
print(f"PREDICTING CONSUMPTION FOR {TARGET_DATE}")
print("="*70)

# Predict on training data
pred_train = predict(X_train_nn, y_train_nn, parameters)
y_pred_train_inv = scaler_y.inverse_transform(pred_train.flatten().reshape(-1, 1)).flatten()
y_actual_train_inv = y_train

# Predict on January 19th
pred_test = predict(X_test_nn, y_test_nn, parameters)
y_pred_test_inv = scaler_y.inverse_transform(pred_test.flatten().reshape(-1, 1)).flatten()
y_actual_test_inv = y_test

# Calculate metrics for TRAIN set
mae_train = mean_absolute_error(y_actual_train_inv, y_pred_train_inv)
mse_train = mean_squared_error(y_actual_train_inv, y_pred_train_inv)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_actual_train_inv, y_pred_train_inv)

# Calculate metrics for TEST set (January 19th)
mae_test = mean_absolute_error(y_actual_test_inv, y_pred_test_inv)
mse_test = mean_squared_error(y_actual_test_inv, y_pred_test_inv)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_actual_test_inv, y_pred_test_inv)

# 8 - PRINT RESULTS
print("\n" + "="*70)
print("RESULTS - CONSUMPTION FORECASTING (5-min resolution)")
print("="*70)
print(f"{'Metric':<12} {'Train':>14} {'Jan 19 Test':>14} {'Diff':>14}")
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

# 9 - CREATE DETAILED 5-MIN PREDICTION TABLE FOR JANUARY 19TH
print("\n" + "="*70)
print(f"5-MINUTE CONSUMPTION FORECAST FOR {TARGET_DATE}")
print("="*70)

results_df = pd.DataFrame({
    'DateTime': timestamps_test,
    'Time': [ts.strftime('%H:%M') for ts in timestamps_test],
    'Actual_Consumption_kW': y_actual_test_inv,
    'Predicted_Consumption_kW': y_pred_test_inv,
    'Error_kW': y_pred_test_inv - y_actual_test_inv,
    'Error_%': ((y_pred_test_inv - y_actual_test_inv) / y_actual_test_inv * 100)
})

print(results_df.to_string(index=False))

# Save results to CSV
results_filename = f'Cons_DeepNN_Prediction_{TARGET_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nResults saved to: {results_filename}")

# 10 - PLOT JANUARY 19TH PREDICTIONS VS ACTUAL
print("\nGenerating prediction plot for January 19th...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Create x-axis values (0 to 288 for 5-min intervals)
x_vals = np.arange(len(timestamps_test))

# Plot 1: Actual vs Predicted
ax1.plot(x_vals, y_actual_test_inv, 'b-', label='Actual Consumption', linewidth=1.5)
ax1.plot(x_vals, y_pred_test_inv, 'r--', label='Predicted Consumption', linewidth=1.5)
ax1.fill_between(x_vals, y_actual_test_inv, y_pred_test_inv, alpha=0.3, color='gray')

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Consumption (kW)', fontsize=12)
ax1.set_title(f'Consumption Forecast for {TARGET_DATE} (5-min resolution)\nR²: {r2_test:.4f}, RMSE: {rmse_test:.2f} kW, MAE: {mae_test:.2f} kW',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# Set x-axis labels every hour (every 12 intervals for 5-min data)
tick_positions = np.arange(0, len(timestamps_test), 12)
tick_labels = [timestamps_test[i].strftime('%H:%M') for i in tick_positions if i < len(timestamps_test)]
ax1.set_xticks(tick_positions[:len(tick_labels)])
ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

# Plot 2: Prediction Error
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

plt.tight_layout()
plot_filename = f'Cons_DeepNN_Prediction_{TARGET_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Prediction plot saved to: {plot_filename}")
plt.show()

# 11 - FINAL SUMMARY
print("\n" + "#"*70)
print("### FINAL SUMMARY - CONSUMPTION FORECASTING (5-min resolution)")
print("#"*70)

print(f"\nTarget Date: {TARGET_DATE}")
print(f"Resolution: 5 minutes")
print(f"Training Period: All data except {TARGET_DATE}")
print(f"Training Samples: {len(y_train)}")
print(f"Prediction Intervals: {len(y_test)}")

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
print("### FORECASTING COMPLETE!")
print("#"*70)
