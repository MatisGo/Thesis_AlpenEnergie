"""
DeepNN Consumption Forecasting with Rolling Cross-Validation
=============================================================
This script implements a robust validation system using Rolling Window Cross-Validation
(also known as Walk-Forward Validation) for time series forecasting.

WHY ROLLING CROSS-VALIDATION?
-----------------------------
Standard cross-validation randomly shuffles data, which causes data leakage in time series
because future data can "leak" into training. Rolling CV respects temporal order:
- Training data always comes BEFORE test data
- Multiple test folds give robust performance estimates
- Prevents overfitting to a single train/test split

VALIDATION STRATEGY:
-------------------
1. EXPANDING WINDOW: Training set grows with each fold
2. MULTIPLE TEST DAYS: Test on N consecutive days (not just one)
3. GAP/PURGING: Optional gap between train and test to reduce autocorrelation
4. FINAL MODEL: Train on all data except final test day for production use

References:
- Hyndman & Athanasopoulos: Forecasting Principles and Practice (fpp3)
- scikit-learn TimeSeriesSplit
- Walk-Forward Validation best practices
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

# Import from parent directory
import sys
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dnn_app_utils_v3 import *

# ========================================================================
# HYPERPARAMETERS - EASY TO CHANGE
# ========================================================================
LEARNING_RATE = 0.6                    # Learning rate for gradient descent
NUM_ITERATIONS = 3000                  # Number of iterations for training
LOOKBACK_STEPS = 288                   # 288 x 5min = 24 hours of past data
PRINT_COST = False                     # Print cost during training
SAVE_MODEL = True                      # Save model for real forecasting

# ========================================================================
# CROSS-VALIDATION CONFIGURATION
# ========================================================================
CV_TEST_DAYS = ['2026-01-23', '2026-01-24', '2026-01-25', '2026-01-26', '2026-01-27']  # Days to test on
CV_GAP_HOURS = 0                       # Gap between train and test (hours) to reduce autocorrelation
CV_MIN_TRAIN_DAYS = 5                  # Minimum training days before first test
FINAL_TEST_DATE = '2026-01-27'         # Final test date for production model

# ========================================================================
# NETWORK ARCHITECTURE
# ========================================================================
LAYERS_DIMS_CONFIG = [128, 64, 32, 16, 1]  # Hidden layers + output (input size added dynamically)

# ========================================================================

print("="*70)
print("DEEPNN CONSUMPTION FORECASTING WITH ROLLING CROSS-VALIDATION")
print("="*70)
print(f"Test days: {CV_TEST_DAYS}")
print(f"Gap between train/test: {CV_GAP_HOURS} hours")
print(f"Minimum training days: {CV_MIN_TRAIN_DAYS}")
print("="*70)

# ============================================================================
# 1 - LOAD AND PREPROCESS DATA
# ============================================================================
print("\n" + "="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)

# Load data from CSV (skip first 3 header rows)
data = pd.read_csv('../Data_January.csv', skiprows=3, header=None, encoding='latin-1')
print(f"Loaded data shape: {data.shape}")

# Assign column names based on the file structure
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

# Add date features
data['Date'] = data['DateTime'].dt.date
data['Hour'] = data['DateTime'].dt.hour
data['Minute'] = data['DateTime'].dt.minute
data['DayOfWeek'] = data['DateTime'].dt.dayofweek
data['Month'] = data['DateTime'].dt.month
data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

print(f"Data shape: {data.shape}")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

# Get list of unique dates
unique_dates = sorted(data['Date'].unique())
print(f"Total days in dataset: {len(unique_dates)}")
print(f"Available dates: {unique_dates[0]} to {unique_dates[-1]}")

# ============================================================================
# 2 - DEFINE FEATURE CREATION FUNCTION
# ============================================================================

def create_forecast_features(df, lookback_steps=288, forecast_step=1):
    """
    Create features for forecasting consumption at a specific step ahead.

    Features include:
    - Past consumption (lookback_steps values)
    - Past temperature (lookback_steps values)
    - Past predicted temperature (lookback_steps values)
    - Future predicted temperature (1 value)
    - Time features (hour, minute, day of week, month, is_weekend)
    """
    features = []
    targets = []
    timestamps = []

    for i in range(lookback_steps, len(df) - forecast_step):
        feature_row = []

        # Past consumption
        past_consumption = df['Consumption'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_consumption)

        # Past temperature
        past_temperature = df['Temperature'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_temperature)

        # Past predicted temperature
        past_temp_pred = df['Temperature_Predicted'].iloc[i-lookback_steps:i].values
        feature_row.extend(past_temp_pred)

        # Future predicted temperature
        future_temp_pred = df['Temperature_Predicted'].iloc[i + forecast_step]
        feature_row.append(future_temp_pred)

        # Time features for target step
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


def train_and_evaluate(X_train, y_train, X_test, y_test, layers_dims,
                       learning_rate, num_iterations, print_cost=False):
    """
    Train the DeepNN model and evaluate on test set.
    Returns predictions, metrics, and trained parameters.
    """
    # Normalize features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Normalize target
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Reshape for neural network (features, samples)
    X_train_nn = X_train_scaled.T
    y_train_nn = y_train_scaled.reshape(1, -1)

    # Train model
    np.random.seed(1)
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X_train_nn, parameters)
        cost = compute_cost(AL, y_train_nn)
        grads = L_model_backward(AL, y_train_nn, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

    # Predict on test set
    X_test_nn = X_test_scaled.T
    dummy_y = np.zeros((1, X_test_nn.shape[1]))
    pred_test = predict(X_test_nn, dummy_y, parameters)
    y_pred = scaler_y.inverse_transform(pred_test.flatten().reshape(-1, 1)).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

    return y_pred, metrics, parameters, scaler_X, scaler_y


# ============================================================================
# 3 - ROLLING CROSS-VALIDATION
# ============================================================================
print("\n" + "="*70)
print("ROLLING CROSS-VALIDATION")
print("="*70)

cv_results = []
all_predictions = []
all_actuals = []
all_timestamps = []
fold_metrics = []

# Convert test days to date objects
test_dates = [pd.to_datetime(d).date() for d in CV_TEST_DAYS]

# Calculate days needed for lookback
days_for_lookback = int(np.ceil(LOOKBACK_STEPS / 288)) + 1

print(f"\nStarting {len(test_dates)}-fold rolling cross-validation...")
print(f"Days needed for lookback features: {days_for_lookback}")
print("-" * 70)

for fold_idx, test_date in enumerate(test_dates):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/{len(test_dates)}: Testing on {test_date}")
    print("="*70)

    # Define train cutoff (with optional gap)
    gap_timedelta = pd.Timedelta(hours=CV_GAP_HOURS)
    train_cutoff = pd.to_datetime(test_date) - gap_timedelta

    # Get training data: all complete days before test date (minus gap)
    train_data = data[data['DateTime'] < train_cutoff].copy()

    # Check if we have enough training data
    train_days = len(train_data['Date'].unique())
    if train_days < CV_MIN_TRAIN_DAYS + days_for_lookback:
        print(f"  SKIPPING: Only {train_days} training days available (need {CV_MIN_TRAIN_DAYS + days_for_lookback})")
        continue

    print(f"  Training data: {len(train_data)} samples ({train_days} days)")
    print(f"  Training period: {train_data['DateTime'].min()} to {train_data['DateTime'].max()}")

    # Get test data for the specific test date
    test_data = data[data['Date'] == test_date].copy()
    print(f"  Test data: {len(test_data)} samples")

    if len(test_data) < 288:  # Less than a full day
        print(f"  SKIPPING: Incomplete test day ({len(test_data)} samples)")
        continue

    # Create features for training
    X_train, y_train, ts_train = create_forecast_features(
        train_data, lookback_steps=LOOKBACK_STEPS, forecast_step=1
    )

    if len(X_train) == 0:
        print(f"  SKIPPING: No training features created")
        continue

    print(f"  Training features: {X_train.shape}")

    # Create features for test day
    # Need lookback data before test date + test date data
    lookback_start = pd.to_datetime(test_date) - pd.Timedelta(days=days_for_lookback)
    combined_data = data[(data['DateTime'] >= lookback_start) &
                         (data['Date'] <= test_date)].copy().reset_index(drop=True)

    # Create test features (only for test date timestamps)
    X_test_list = []
    y_test_list = []
    timestamps_test = []

    for i in range(LOOKBACK_STEPS, len(combined_data)):
        current_datetime = combined_data['DateTime'].iloc[i]
        if current_datetime.date() != test_date:
            continue

        feature_row = []

        # Past features
        past_consumption = combined_data['Consumption'].iloc[i-LOOKBACK_STEPS:i].values
        feature_row.extend(past_consumption)

        past_temperature = combined_data['Temperature'].iloc[i-LOOKBACK_STEPS:i].values
        feature_row.extend(past_temperature)

        past_temp_pred = combined_data['Temperature_Predicted'].iloc[i-LOOKBACK_STEPS:i].values
        feature_row.extend(past_temp_pred)

        # Future temp
        feature_row.append(combined_data['Temperature_Predicted'].iloc[i])

        # Time features
        feature_row.append(current_datetime.hour)
        feature_row.append(current_datetime.minute)
        feature_row.append(current_datetime.dayofweek)
        feature_row.append(current_datetime.month)
        feature_row.append(1 if current_datetime.dayofweek >= 5 else 0)

        X_test_list.append(feature_row)
        y_test_list.append(combined_data['Consumption'].iloc[i])
        timestamps_test.append(current_datetime)

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    if len(X_test) == 0:
        print(f"  SKIPPING: No test features created")
        continue

    print(f"  Test features: {X_test.shape}")

    # Define network architecture
    n_features = X_train.shape[1]
    layers_dims = [n_features] + LAYERS_DIMS_CONFIG

    # Train and evaluate
    start_time = time.time()
    y_pred, metrics, parameters, scaler_X, scaler_y = train_and_evaluate(
        X_train, y_train, X_test, y_test, layers_dims,
        LEARNING_RATE, NUM_ITERATIONS, PRINT_COST
    )
    fold_time = time.time() - start_time

    # Store results
    fold_result = {
        'fold': fold_idx + 1,
        'test_date': test_date,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'r2': metrics['r2'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'time': fold_time
    }
    cv_results.append(fold_result)
    fold_metrics.append(metrics)

    # Store predictions
    all_predictions.extend(y_pred)
    all_actuals.extend(y_test)
    all_timestamps.extend(timestamps_test)

    # Print fold results
    print(f"\n  Results for {test_date}:")
    print(f"    R²:   {metrics['r2']:.4f}")
    print(f"    MAE:  {metrics['mae']:.2f} kW")
    print(f"    RMSE: {metrics['rmse']:.2f} kW")
    print(f"    MAPE: {metrics['mape']:.2f}%")
    print(f"    Time: {fold_time:.2f}s")

# ============================================================================
# 4 - CROSS-VALIDATION SUMMARY
# ============================================================================
print("\n" + "="*70)
print("CROSS-VALIDATION SUMMARY")
print("="*70)

if len(cv_results) > 0:
    cv_df = pd.DataFrame(cv_results)

    print(f"\n{'Fold':<6} {'Test Date':<12} {'Train':<8} {'Test':<6} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    print("-" * 80)

    for _, row in cv_df.iterrows():
        print(f"{row['fold']:<6} {str(row['test_date']):<12} {row['train_samples']:<8} {row['test_samples']:<6} "
              f"{row['r2']:>8.4f} {row['mae']:>10.2f} {row['rmse']:>10.2f} {row['mape']:>10.2f}%")

    print("-" * 80)
    print(f"{'MEAN':<6} {'':<12} {'':<8} {'':<6} "
          f"{cv_df['r2'].mean():>8.4f} {cv_df['mae'].mean():>10.2f} {cv_df['rmse'].mean():>10.2f} {cv_df['mape'].mean():>10.2f}%")
    print(f"{'STD':<6} {'':<12} {'':<8} {'':<6} "
          f"{cv_df['r2'].std():>8.4f} {cv_df['mae'].std():>10.2f} {cv_df['rmse'].std():>10.2f} {cv_df['mape'].std():>10.2f}%")
    print(f"{'MIN':<6} {'':<12} {'':<8} {'':<6} "
          f"{cv_df['r2'].min():>8.4f} {cv_df['mae'].min():>10.2f} {cv_df['rmse'].min():>10.2f} {cv_df['mape'].min():>10.2f}%")
    print(f"{'MAX':<6} {'':<12} {'':<8} {'':<6} "
          f"{cv_df['r2'].max():>8.4f} {cv_df['mae'].max():>10.2f} {cv_df['rmse'].max():>10.2f} {cv_df['mape'].max():>10.2f}%")

    # Save CV results
    cv_df.to_csv('Cons_DeepNN_CV_Results.csv', index=False)
    print(f"\nCV results saved to: Cons_DeepNN_CV_Results.csv")
else:
    print("No cross-validation results to summarize!")

# ============================================================================
# 5 - TRAIN FINAL MODEL ON ALL DATA (except final test date)
# ============================================================================
print("\n" + "="*70)
print(f"TRAINING FINAL MODEL (Test: {FINAL_TEST_DATE})")
print("="*70)

final_test_date = pd.to_datetime(FINAL_TEST_DATE).date()

# Training data: all data before final test date
train_data_final = data[data['Date'] < final_test_date].copy()
test_data_final = data[data['Date'] == final_test_date].copy()

print(f"Final training data: {len(train_data_final)} samples")
print(f"Final test data: {len(test_data_final)} samples")

# Create training features
X_train_final, y_train_final, _ = create_forecast_features(
    train_data_final, lookback_steps=LOOKBACK_STEPS, forecast_step=1
)

# Create test features
lookback_start = pd.to_datetime(FINAL_TEST_DATE) - pd.Timedelta(days=days_for_lookback)
combined_final = data[(data['DateTime'] >= lookback_start) &
                      (data['Date'] <= final_test_date)].copy().reset_index(drop=True)

X_test_final_list = []
y_test_final_list = []
timestamps_final = []

for i in range(LOOKBACK_STEPS, len(combined_final)):
    current_datetime = combined_final['DateTime'].iloc[i]
    if current_datetime.date() != final_test_date:
        continue

    feature_row = []
    feature_row.extend(combined_final['Consumption'].iloc[i-LOOKBACK_STEPS:i].values)
    feature_row.extend(combined_final['Temperature'].iloc[i-LOOKBACK_STEPS:i].values)
    feature_row.extend(combined_final['Temperature_Predicted'].iloc[i-LOOKBACK_STEPS:i].values)
    feature_row.append(combined_final['Temperature_Predicted'].iloc[i])
    feature_row.append(current_datetime.hour)
    feature_row.append(current_datetime.minute)
    feature_row.append(current_datetime.dayofweek)
    feature_row.append(current_datetime.month)
    feature_row.append(1 if current_datetime.dayofweek >= 5 else 0)

    X_test_final_list.append(feature_row)
    y_test_final_list.append(combined_final['Consumption'].iloc[i])
    timestamps_final.append(current_datetime)

X_test_final = np.array(X_test_final_list)
y_test_final = np.array(y_test_final_list)

# Define architecture
n_features = X_train_final.shape[1]
LAYERS_DIMS = [n_features] + LAYERS_DIMS_CONFIG

print(f"Network architecture: {LAYERS_DIMS}")
print(f"Training features: {X_train_final.shape}")
print(f"Test features: {X_test_final.shape}")

# Train final model
start_time = time.time()
y_pred_final, metrics_final, parameters_final, scaler_X_final, scaler_y_final = train_and_evaluate(
    X_train_final, y_train_final, X_test_final, y_test_final, LAYERS_DIMS,
    LEARNING_RATE, NUM_ITERATIONS, PRINT_COST
)
final_time = time.time() - start_time

print(f"\nFinal Model Results on {FINAL_TEST_DATE}:")
print(f"  R²:   {metrics_final['r2']:.4f}")
print(f"  MAE:  {metrics_final['mae']:.2f} kW")
print(f"  RMSE: {metrics_final['rmse']:.2f} kW")
print(f"  MAPE: {metrics_final['mape']:.2f}%")
print(f"  Time: {final_time:.2f}s")

# ============================================================================
# 6 - SAVE DETAILED RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save final test day predictions
results_df = pd.DataFrame({
    'DateTime': timestamps_final,
    'Time': [ts.strftime('%H:%M') for ts in timestamps_final],
    'Actual_Consumption_kW': y_test_final,
    'Predicted_Consumption_kW': y_pred_final,
    'Error_kW': y_pred_final - y_test_final,
    'Error_%': ((y_pred_final - y_test_final) / y_test_final * 100)
})

results_filename = f'Cons_DeepNN_Prediction_{FINAL_TEST_DATE.replace("-", "")}_5min.csv'
results_df.to_csv(results_filename, index=False)
print(f"Predictions saved to: {results_filename}")

# ============================================================================
# 7 - VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(18, 16))

# Plot 1: Cross-validation metrics across folds
ax1 = axes[0, 0]
if len(cv_results) > 0:
    folds = [r['fold'] for r in cv_results]
    r2_scores = [r['r2'] for r in cv_results]
    ax1.bar(folds, r2_scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=np.mean(r2_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_scores):.4f}')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Cross-Validation R² Scores by Fold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(folds)
    ax1.set_xticklabels([str(r['test_date']) for r in cv_results], rotation=45, ha='right')

# Plot 2: MAE/RMSE across folds
ax2 = axes[0, 1]
if len(cv_results) > 0:
    x = np.arange(len(cv_results))
    width = 0.35
    mae_scores = [r['mae'] for r in cv_results]
    rmse_scores = [r['rmse'] for r in cv_results]
    ax2.bar(x - width/2, mae_scores, width, label='MAE', color='steelblue', alpha=0.7)
    ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='coral', alpha=0.7)
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Error (kW)', fontsize=12)
    ax2.set_title('MAE and RMSE by Fold', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(r['test_date']) for r in cv_results], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Final test day - Actual vs Predicted
ax3 = axes[1, 0]
x_vals = np.arange(len(timestamps_final))
ax3.plot(x_vals, y_test_final, 'b-', label='Actual', linewidth=1.5)
ax3.plot(x_vals, y_pred_final, 'r--', label='Predicted', linewidth=1.5)
ax3.fill_between(x_vals, y_test_final, y_pred_final, alpha=0.3, color='gray')
ax3.set_xlabel('Time of Day', fontsize=12)
ax3.set_ylabel('Consumption (kW)', fontsize=12)
ax3.set_title(f'Final Model: {FINAL_TEST_DATE}\nR²: {metrics_final["r2"]:.4f}, RMSE: {metrics_final["rmse"]:.2f} kW',
              fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
tick_positions = np.arange(0, len(timestamps_final), 24)
tick_labels = [timestamps_final[i].strftime('%H:%M') for i in tick_positions if i < len(timestamps_final)]
ax3.set_xticks(tick_positions[:len(tick_labels)])
ax3.set_xticklabels(tick_labels, rotation=0)

# Plot 4: Prediction errors for final test day
ax4 = axes[1, 1]
errors = y_pred_final - y_test_final
colors = ['green' if e < 0 else 'red' for e in errors]
ax4.bar(x_vals, errors, color=colors, alpha=0.7, width=1.0)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.axhline(y=metrics_final['mae'], color='red', linestyle='--', label=f'+MAE: {metrics_final["mae"]:.2f}')
ax4.axhline(y=-metrics_final['mae'], color='red', linestyle='--', label=f'-MAE')
ax4.set_xlabel('Time of Day', fontsize=12)
ax4.set_ylabel('Error (kW)', fontsize=12)
ax4.set_title('Prediction Errors (Predicted - Actual)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks(tick_positions[:len(tick_labels)])
ax4.set_xticklabels(tick_labels, rotation=0)

# Plot 5: All CV predictions combined (scatter)
ax5 = axes[2, 0]
if len(all_actuals) > 0:
    ax5.scatter(all_actuals, all_predictions, alpha=0.3, s=10, c='steelblue')
    min_val = min(min(all_actuals), min(all_predictions))
    max_val = max(max(all_actuals), max(all_predictions))
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    overall_r2 = r2_score(all_actuals, all_predictions)
    ax5.set_xlabel('Actual Consumption (kW)', fontsize=12)
    ax5.set_ylabel('Predicted Consumption (kW)', fontsize=12)
    ax5.set_title(f'All CV Predictions vs Actual\nOverall R²: {overall_r2:.4f}', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

# Plot 6: Error distribution
ax6 = axes[2, 1]
if len(all_actuals) > 0:
    all_errors = np.array(all_predictions) - np.array(all_actuals)
    ax6.hist(all_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax6.axvline(x=np.mean(all_errors), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(all_errors):.2f} kW')
    ax6.set_xlabel('Prediction Error (kW)', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Distribution of Prediction Errors (All CV Folds)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_filename = f'Cons_DeepNN_CV_Results.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_filename}")
plt.show()

# ============================================================================
# 8 - SAVE FINAL MODEL
# ============================================================================
if SAVE_MODEL:
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)

    model_filename = 'Cons_DeepNN_Model.npz'

    np.savez(model_filename,
             # Model parameters
             **{f'W{l}': parameters_final[f'W{l}'] for l in range(1, len(LAYERS_DIMS))},
             **{f'b{l}': parameters_final[f'b{l}'] for l in range(1, len(LAYERS_DIMS))},
             # Architecture
             layers_dims=np.array(LAYERS_DIMS),
             lookback_steps=np.array(LOOKBACK_STEPS),
             # Scalers
             scaler_X_min=scaler_X_final.data_min_,
             scaler_X_max=scaler_X_final.data_max_,
             scaler_X_scale=scaler_X_final.scale_,
             scaler_X_data_range=scaler_X_final.data_range_,
             scaler_y_min=scaler_y_final.data_min_,
             scaler_y_max=scaler_y_final.data_max_,
             scaler_y_scale=scaler_y_final.scale_,
             scaler_y_data_range=scaler_y_final.data_range_,
             # CV metrics
             cv_mean_r2=np.array(cv_df['r2'].mean() if len(cv_results) > 0 else 0),
             cv_mean_mae=np.array(cv_df['mae'].mean() if len(cv_results) > 0 else 0),
             cv_mean_rmse=np.array(cv_df['rmse'].mean() if len(cv_results) > 0 else 0),
             cv_std_r2=np.array(cv_df['r2'].std() if len(cv_results) > 0 else 0),
             # Final test metrics
             training_r2=np.array(metrics_final['r2']),
             training_mae=np.array(metrics_final['mae']),
             training_rmse=np.array(metrics_final['rmse'])
    )

    print(f"Model saved to: {model_filename}")
    print(f"  - Architecture: {LAYERS_DIMS}")
    print(f"  - Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
    print(f"  - CV Mean R²: {cv_df['r2'].mean():.4f} (±{cv_df['r2'].std():.4f})" if len(cv_results) > 0 else "")
    print(f"  - Final Test R²: {metrics_final['r2']:.4f}")

# ============================================================================
# 9 - FINAL SUMMARY
# ============================================================================
print("\n" + "#"*70)
print("### FINAL SUMMARY - ROLLING CROSS-VALIDATION")
print("#"*70)

print(f"\nValidation Strategy: Rolling Window Cross-Validation")
print(f"Number of Folds: {len(cv_results)}")
print(f"Test Days: {CV_TEST_DAYS}")
print(f"Gap Between Train/Test: {CV_GAP_HOURS} hours")

if len(cv_results) > 0:
    print(f"\n--- Cross-Validation Performance ---")
    print(f"Mean R²:   {cv_df['r2'].mean():.4f} (±{cv_df['r2'].std():.4f})")
    print(f"Mean MAE:  {cv_df['mae'].mean():.2f} kW (±{cv_df['mae'].std():.2f})")
    print(f"Mean RMSE: {cv_df['rmse'].mean():.2f} kW (±{cv_df['rmse'].std():.2f})")
    print(f"Mean MAPE: {cv_df['mape'].mean():.2f}% (±{cv_df['mape'].std():.2f})")

print(f"\n--- Final Model ({FINAL_TEST_DATE}) ---")
print(f"R²:   {metrics_final['r2']:.4f}")
print(f"MAE:  {metrics_final['mae']:.2f} kW")
print(f"RMSE: {metrics_final['rmse']:.2f} kW")
print(f"MAPE: {metrics_final['mape']:.2f}%")

print(f"\n--- Model Configuration ---")
print(f"Architecture: {LAYERS_DIMS}")
print(f"Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Iterations: {NUM_ITERATIONS}")

print("\n" + "#"*70)
print("### CROSS-VALIDATION COMPLETE!")
print("#"*70)
