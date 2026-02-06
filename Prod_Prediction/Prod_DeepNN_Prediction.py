"""
DeepNN-based Production Forecasting with Lookback Features
==========================================================
This script uses a Deep Neural Network for solar/hydro production forecasting.

KEY FEATURES FOR PRODUCTION PREDICTION:
---------------------------------------
Based on research (2024-2025), the best approaches for production forecasting include:
1. Irradiance as the primary feature (most important for solar)
2. Temperature (affects PV efficiency)
3. Water levels (for hydro component)
4. Time features (hour, day of year for seasonal patterns)
5. Historical production patterns (lookback)

Architecture based on CNN-LSTM research showing R² > 0.97 for PV forecasting.
For feedforward DNN, we use deep architecture with dropout-like regularization.

Reference: https://www.nature.com/articles/s41598-025-14908-x
"""

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
LEARNING_RATE = 0.5                                   # Learning rate for gradient descent
NUM_ITERATIONS = 3000                                 # Number of iterations for training
LOOKBACK_STEPS = 288                                  # 288 x 5min = 24 hours of past data
PRINT_COST = False                                    # Print cost during training
SAVE_MODEL = True                                     # Save model for real forecasting
# ========================================================================
# TARGET DATE TO PREDICT (excluded from training)
TARGET_DATE = '2025-01-27'
# ========================================================================

# 1 - LOAD AND PREPROCESS DATA
print("="*70)
print("LOADING AND PREPROCESSING DATA FOR PRODUCTION FORECASTING")
print(f"Target prediction date: {TARGET_DATE}")
print(f"Resolution: 5 minutes")
print("="*70)

# Load data from CSV
data = pd.read_csv('matis_2025_.csv')
print(f"Loaded data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Parse DateTime
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Daytime'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
data = data.dropna(subset=['DateTime'])
data = data.sort_values('DateTime').reset_index(drop=True)

print(f"Data shape after parsing: {data.shape}")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

# Rename columns for consistency
data.rename(columns={
    'Haselholtz Water level': 'Level_Haselholz',
    'Bidmi Water level': 'Level_Bidmi'
}, inplace=True)

# Extract time features
data['Hour'] = data['DateTime'].dt.hour
data['Minute'] = data['DateTime'].dt.minute
data['DayOfWeek'] = data['DateTime'].dt.dayofweek
data['Month'] = data['DateTime'].dt.month
data['DayOfYear'] = data['DateTime'].dt.dayofyear
data['Date_only'] = data['DateTime'].dt.date

# Create cyclical time features (important for production patterns)
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['DayOfYear_sin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
data['DayOfYear_cos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)

print(f"\n5-minute data preview:")
print(data.head(10))

# 2 - SPLIT DATA: EXCLUDE TARGET DATE FROM TRAINING
print("\n" + "="*70)
print(f"SPLITTING DATA - EXCLUDING {TARGET_DATE} FOR PREDICTION")
print("="*70)

target_date = pd.to_datetime(TARGET_DATE).date()

# Training data: all days BEFORE target date
train_data = data[data['Date_only'] < target_date].copy()

# Test data: target date only
test_data = data[data['Date_only'] == target_date].copy()

print(f"Training data: {len(train_data)} intervals (5-min)")
print(f"Test data ({TARGET_DATE}): {len(test_data)} intervals (5-min)")

# Lookback data for test predictions
days_needed = int(np.ceil(LOOKBACK_STEPS / 288)) + 1
lookback_start = pd.to_datetime(TARGET_DATE) - pd.Timedelta(days=days_needed)
lookback_data = data[(data['DateTime'] >= lookback_start) &
                     (data['Date_only'] < target_date)].copy()
print(f"Lookback data: {len(lookback_data)} intervals (5-min) from {days_needed} days before")

# 3 - CREATE FEATURES FOR TRAINING
print("\n" + "="*70)
print("CREATING FEATURES FOR TRAINING")
print("="*70)

def create_production_features(df, lookback_steps=288, forecast_step=1):
    """
    Create features for forecasting production at a specific step ahead.

    Features include:
    - Past production values (most important)
    - Past irradiance (key for solar)
    - Past temperature (affects PV efficiency)
    - Past water levels (for hydro)
    - Time features
    """
    features = []
    targets = []
    timestamps = []

    for i in range(lookback_steps, len(df) - forecast_step):
        feature_row = []

        # Past production (last 24 hours = 288 x 5min)
        past_production = df['Production'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_production)

        # Past irradiance (key feature for solar production)
        past_irradiance = df['Irradiance'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_irradiance)

        # Past temperature (affects PV efficiency - negative correlation)
        past_temperature = df['Temperature'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_temperature)

        # Past water levels (for hydro component)
        past_level_bidmi = df['Level_Bidmi'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_level_bidmi)

        past_level_haselholz = df['Level_Haselholz'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_level_haselholz)

        # Current irradiance at target step (if available as forecast)
        # This simulates having an irradiance forecast
        future_irradiance = df['Irradiance'].iloc[i + forecast_step]
        feature_row.append(future_irradiance)

        # Time features for the target step
        target_datetime = df['DateTime'].iloc[i + forecast_step]
        feature_row.append(np.sin(2 * np.pi * target_datetime.hour / 24))  # Hour_sin
        feature_row.append(np.cos(2 * np.pi * target_datetime.hour / 24))  # Hour_cos
        feature_row.append(np.sin(2 * np.pi * target_datetime.dayofyear / 365))  # DayOfYear_sin
        feature_row.append(np.cos(2 * np.pi * target_datetime.dayofyear / 365))  # DayOfYear_cos
        feature_row.append(target_datetime.month)

        features.append(feature_row)
        targets.append(df['Production'].iloc[i + forecast_step])
        timestamps.append(target_datetime)

    return np.array(features), np.array(targets), timestamps

# Create training features
print(f"Creating training features with {LOOKBACK_STEPS} steps lookback ({LOOKBACK_STEPS * 5 / 60:.1f} hours)...")
X_train, y_train, timestamps_train = create_production_features(train_data, lookback_steps=LOOKBACK_STEPS, forecast_step=1)

# Define feature count
n_lookback_features = LOOKBACK_STEPS * 5  # Production, Irradiance, Temperature, Level_Bidmi, Level_Haselholz
n_extra_features = 6  # Future_irradiance, Hour_sin, Hour_cos, DayOfYear_sin, DayOfYear_cos, Month
total_features = n_lookback_features + n_extra_features

print(f"Total features: {total_features}")
print(f"  - Lookback features: {n_lookback_features} ({LOOKBACK_STEPS} steps x 5 variables)")
print(f"  - Extra features: {n_extra_features}")
print(f"Training feature matrix shape: {X_train.shape}")
print(f"Training target vector shape: {y_train.shape}")

# 4 - CREATE FEATURES FOR TARGET DATE PREDICTION
print("\n" + "="*70)
print(f"CREATING FEATURES FOR {TARGET_DATE} PREDICTION")
print("="*70)

# Combine lookback data with test data
combined_data = pd.concat([lookback_data, test_data]).reset_index(drop=True)
print(f"Combined data for prediction: {len(combined_data)} intervals (5-min)")

# Create test features
X_test_list = []
y_test_list = []
timestamps_test = []

for i in range(LOOKBACK_STEPS, len(combined_data)):
    current_datetime = combined_data['DateTime'].iloc[i]
    if current_datetime.date() != target_date:
        continue

    feature_row = []

    # Past production
    past_production = combined_data['Production'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_production)

    # Past irradiance
    past_irradiance = combined_data['Irradiance'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_irradiance)

    # Past temperature
    past_temperature = combined_data['Temperature'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_temperature)

    # Past water levels
    past_level_bidmi = combined_data['Level_Bidmi'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_level_bidmi)

    past_level_haselholz = combined_data['Level_Haselholz'].iloc[i-LOOKBACK_STEPS:i].values
    feature_row.extend(past_level_haselholz)

    # Future irradiance (using actual value - in real forecast this would be a forecast)
    future_irradiance = combined_data['Irradiance'].iloc[i]
    feature_row.append(future_irradiance)

    # Time features
    feature_row.append(np.sin(2 * np.pi * current_datetime.hour / 24))
    feature_row.append(np.cos(2 * np.pi * current_datetime.hour / 24))
    feature_row.append(np.sin(2 * np.pi * current_datetime.dayofyear / 365))
    feature_row.append(np.cos(2 * np.pi * current_datetime.dayofyear / 365))
    feature_row.append(current_datetime.month)

    X_test_list.append(feature_row)
    y_test_list.append(combined_data['Production'].iloc[i])
    timestamps_test.append(current_datetime)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

print(f"Test feature matrix shape: {X_test.shape}")
print(f"Test target vector shape: {y_test.shape}")
print(f"5-min intervals to predict: {len(timestamps_test)}")

# 5 - NORMALIZE AND PREPARE FOR NEURAL NETWORK
print("\n" + "="*70)
print("NORMALIZING DATA")
print("="*70)

# Network architecture - deep network for complex patterns
n_features = X_train.shape[1]
# Architecture inspired by successful PV forecasting research
# Deeper networks with decreasing width capture hierarchical patterns
LAYERS_DIMS = [n_features, 256, 128, 64, 32, 16, 1]
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
print(f"Test samples ({TARGET_DATE}): {X_test_nn.shape[1]}")

# 6 - TRAIN THE MODEL
print("\n" + "="*70)
print("TRAINING MODEL FOR PRODUCTION FORECASTING")
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

# 7 - MAKE PREDICTIONS
print("\n" + "="*70)
print(f"PREDICTING PRODUCTION FOR {TARGET_DATE}")
print("="*70)

# Predict on training data
pred_train = predict(X_train_nn, y_train_nn, parameters)
y_pred_train_inv = scaler_y.inverse_transform(pred_train.flatten().reshape(-1, 1)).flatten()
y_actual_train_inv = y_train

# Predict on test data
pred_test = predict(X_test_nn, y_test_nn, parameters)
y_pred_test_inv = scaler_y.inverse_transform(pred_test.flatten().reshape(-1, 1)).flatten()
y_actual_test_inv = y_test

# Calculate metrics for TRAIN set
mae_train = mean_absolute_error(y_actual_train_inv, y_pred_train_inv)
mse_train = mean_squared_error(y_actual_train_inv, y_pred_train_inv)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_actual_train_inv, y_pred_train_inv)

# Calculate metrics for TEST set
mae_test = mean_absolute_error(y_actual_test_inv, y_pred_test_inv)
mse_test = mean_squared_error(y_actual_test_inv, y_pred_test_inv)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_actual_test_inv, y_pred_test_inv)

# 8 - PRINT RESULTS
print("\n" + "="*70)
print("RESULTS - PRODUCTION FORECASTING (5-min resolution)")
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

# 9 - CREATE DETAILED PREDICTION TABLE
print("\n" + "="*70)
print(f"5-MINUTE PRODUCTION FORECAST FOR {TARGET_DATE}")
print("="*70)

results_df = pd.DataFrame({
    'DateTime': timestamps_test,
    'Time': [ts.strftime('%H:%M') for ts in timestamps_test],
    'Actual_Production_kW': y_actual_test_inv,
    'Predicted_Production_kW': y_pred_test_inv,
    'Error_kW': y_pred_test_inv - y_actual_test_inv,
    'Error_%': ((y_pred_test_inv - y_actual_test_inv) / (y_actual_test_inv + 1e-6) * 100)
})

print(results_df.to_string(index=False))

# Save results to CSV
results_filename = f'Prod_DeepNN_Prediction_{TARGET_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"\nResults saved to: {results_filename}")

# 10 - PLOT PREDICTIONS VS ACTUAL
print("\nGenerating prediction plot...")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

x_vals = np.arange(len(timestamps_test))

# Plot 1: Actual vs Predicted
ax1 = axes[0]
ax1.plot(x_vals, y_actual_test_inv, 'b-', label='Actual Production', linewidth=1.5)
ax1.plot(x_vals, y_pred_test_inv, 'r--', label='Predicted Production', linewidth=1.5)
ax1.fill_between(x_vals, y_actual_test_inv, y_pred_test_inv, alpha=0.3, color='gray')

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Production (kW)', fontsize=12)
ax1.set_title(f'Production Forecast for {TARGET_DATE} (5-min resolution)\nR²: {r2_test:.4f}, RMSE: {rmse_test:.2f} kW, MAE: {mae_test:.2f} kW',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

tick_positions = np.arange(0, len(timestamps_test), 12)
tick_labels = [timestamps_test[i].strftime('%H:%M') for i in tick_positions if i < len(timestamps_test)]
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

# Plot 3: Irradiance overlay (key driver)
ax3 = axes[2]
irradiance_test = combined_data[combined_data['Date_only'] == target_date]['Irradiance'].values[:len(timestamps_test)]
ax3_twin = ax3.twinx()

ax3.plot(x_vals, y_actual_test_inv, 'b-', label='Actual Production', linewidth=1.5)
ax3.plot(x_vals, y_pred_test_inv, 'r--', label='Predicted Production', linewidth=1.5)
if len(irradiance_test) == len(x_vals):
    ax3_twin.plot(x_vals, irradiance_test, 'orange', label='Irradiance', linewidth=1.5, alpha=0.7)
    ax3_twin.set_ylabel('Irradiance (W/m²)', fontsize=12, color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')

ax3.set_xlabel('Time of Day', fontsize=12)
ax3.set_ylabel('Production (kW)', fontsize=12)
ax3.set_title('Production vs Irradiance', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(tick_positions[:len(tick_labels)])
ax3.set_xticklabels(tick_labels, rotation=45, ha='right')

plt.tight_layout()
plot_filename = f'Prod_DeepNN_Prediction_{TARGET_DATE.replace("-", "")}_5min.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Prediction plot saved to: {plot_filename}")
plt.show()

# 11 - FINAL SUMMARY
print("\n" + "#"*70)
print("### FINAL SUMMARY - PRODUCTION FORECASTING (5-min resolution)")
print("#"*70)

print(f"\nTarget Date: {TARGET_DATE}")
print(f"Resolution: 5 minutes")
print(f"Training Period: All data before {TARGET_DATE}")
print(f"Training Samples: {len(y_train)}")
print(f"Prediction Intervals: {len(y_test)}")

print(f"\n--- Model Performance on {TARGET_DATE} ---")
print(f"R²:   {r2_test:.4f}")
print(f"MAE:  {mae_test:.2f} kW")
print(f"RMSE: {rmse_test:.2f} kW")
print(f"MAPE: {np.mean(np.abs(results_df['Error_%'])):.2f}%")

print(f"\n--- Daily Summary ---")
print(f"Total Actual Production:    {y_actual_test_inv.sum() / 12:.2f} kWh")
print(f"Total Predicted Production: {y_pred_test_inv.sum() / 12:.2f} kWh")
daily_error = (y_pred_test_inv.sum() - y_actual_test_inv.sum()) / 12
daily_error_pct = (y_pred_test_inv.sum() - y_actual_test_inv.sum()) / (y_actual_test_inv.sum() + 1e-6) * 100
print(f"Daily Error:                 {daily_error:.2f} kWh ({daily_error_pct:.2f}%)")

print(f"\nComputation Time: {training_time:.2f}s ({training_time/60:.2f} min)")

print("\n" + "#"*70)
print("### PRODUCTION FORECASTING COMPLETE!")
print("#"*70)

# 12 - SAVE MODEL (if enabled)
if SAVE_MODEL:
    print("\n" + "="*70)
    print("SAVING MODEL FOR REAL FORECASTING")
    print("="*70)

    model_filename = 'Prod_DeepNN_Model.npz'

    np.savez(model_filename,
             # Model parameters (weights and biases)
             **{f'W{l}': parameters[f'W{l}'] for l in range(1, len(LAYERS_DIMS))},
             **{f'b{l}': parameters[f'b{l}'] for l in range(1, len(LAYERS_DIMS))},
             # Model architecture
             layers_dims=np.array(LAYERS_DIMS),
             lookback_steps=np.array(LOOKBACK_STEPS),
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
             training_rmse=np.array(rmse_train)
    )

    print(f"Model saved to: {model_filename}")
    print(f"  - Architecture: {LAYERS_DIMS}")
    print(f"  - Lookback steps: {LOOKBACK_STEPS}")
    print(f"  - Training R²: {r2_train:.4f}")
    print("="*70)
