import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dnn_app_utils_v3 import *

# ========================================================================
# HYPERPARAMETERS - EASY TO CHANGE
# ========================================================================
LEARNING_RATE = 0.5                                   # Learning rate for gradient descent (OPTIMAL for this problem)
NUM_ITERATIONS = 3000                                 # Number of iterations for training
LAYERS_DIMS = [9, 9, 12, 15, 9, 1]                      # Network architecture (9 input features)
PRINT_COST = False                                     # Print cost during training (TRUE to see if it's decreasing!)
TEST_SIZE = 0.2                                        # Train/test split ratio
RANDOM_STATE = 42                                      # Random seed for reproducibility
# ========================================================================

# 1 - LOAD AND PREPROCESS DATA
print("="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)

# Load data from CSV
data = pd.read_csv('matis_2025_.csv')
print(f"Loaded data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Drop rows with missing values
data = data.dropna()
print(f"Data shape after dropping NaN: {data.shape}")

# Parse Date column to extract date features
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y', errors='coerce')
data = data.dropna(subset=['Date'])

# Extract month and day of year from Date
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear

# Parse Daytime column to extract hour
data['Hour'] = pd.to_datetime(data['Daytime'], format='%H:%M:%S', errors='coerce').dt.hour
data = data.dropna(subset=['Hour'])

# Define feature columns
# Features: Hour (from Daytime), Rain, Temperature, Consumption, Irradiance,
#           Haselholtz Water level, Bidmi Water level, Month (from Date), DayOfYear
feature_columns = [
    'Hour',
    'Rain',
    'Temperature',
    'Consumption',
    'Irradiance',
    'Haselholtz Water level',
    'Bidmi Water level',
    'Month',
    'DayOfYear'
]

# Target column
target_column = 'Production'

print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")
print(f"Target column: {target_column}")
print(f"\nData preview:")
print(data[feature_columns + [target_column]].head())

# Separate features and target variable
X = data[feature_columns].values.astype(float)
y = data[target_column].values.astype(float)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Normalize features and target
scaler_X = MinMaxScaler(feature_range=(0,1))
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0,1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Reshape for neural network (features, samples)
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)
X_train = X_train.T
X_test = X_test.T

# Print dataset info
m_train = X_train.shape[1]
m_test = X_test.shape[1]
n_x = X_train.shape[0]
n_y = y_train.shape[0]

print("\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)
print(f"Number of training examples: m_train = {m_train}")
print(f"Number of testing examples: m_test = {m_test}")
print(f"Number of features: n_x = {n_x}")
print(f"Number of output units: n_y = {n_y}")
print(f"train_set_x shape: {X_train.shape}")
print(f"train_set_y shape: {y_train.shape}")
print(f"test_set_x shape: {X_test.shape}")
print(f"test_set_y shape: {y_test.shape}")

# 2 - DEFINE L-LAYER MODEL FUNCTION
def L_layer_model(X, Y, layers_dims, learning_rate=0.85, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- list of costs recorded during training
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

# 3 - TRAIN THE MODEL
print("\n" + "="*70)
print("TRAINING MODEL")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Iterations: {NUM_ITERATIONS}")
print(f"Architecture: {LAYERS_DIMS}")
print("="*70)

# Start timing
start_time = time.time()

# Train the model
print(f"Training model with {NUM_ITERATIONS} iterations...")
parameters, costs = L_layer_model(X_train, y_train, LAYERS_DIMS,
                                  learning_rate=LEARNING_RATE,
                                  num_iterations=NUM_ITERATIONS,
                                  print_cost=PRINT_COST)

# End timing
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")

# Make predictions
pred_train = predict(X_train, y_train, parameters)
pred_test = predict(X_test, y_test, parameters)

# Inverse transform to original scale
y_pred_train_inv = scaler_y.inverse_transform(pred_train.flatten().reshape(-1, 1)).flatten()
y_actual_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_pred_test_inv = scaler_y.inverse_transform(pred_test.flatten().reshape(-1, 1)).flatten()
y_actual_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

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

# 4 - PRINT RESULTS
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"{'Metric':<12} {'Train':>14} {'Test':>14} {'Diff':>14}")
print("-" * 70)
print(f"{'R²':<12} {r2_train:>14.6f} {r2_test:>14.6f} {r2_test - r2_train:>14.6f}")
print(f"{'MAE':<12} {mae_train:>14.4f} {mae_test:>14.4f} {mae_test - mae_train:>14.4f}")
print(f"{'MSE':<12} {mse_train:>14.4f} {mse_test:>14.4f} {mse_test - mse_train:>14.4f}")
print(f"{'RMSE':<12} {rmse_train:>14.4f} {rmse_test:>14.4f} {rmse_test - rmse_train:>14.4f}")
print("-" * 70)

# Diagnostic
if r2_train > 0.9 and r2_test < 0.7:
    diagnostic = "OVERFITTING: High Train R², Low Test R²"
elif r2_train < 0.5 and r2_test < 0.5:
    diagnostic = "UNDERFITTING: Poor performance on Train and Test"
elif abs(r2_train - r2_test) < 0.1:
    diagnostic = "GOOD FIT: Similar performance on Train/Test"
else:
    diagnostic = "POSSIBLE OVERFITTING: Notable gap between Train and Test"

print(f"Diagnostic: {diagnostic}")
print(f"Computation Time: {training_time:.2f}s ({training_time/60:.2f} min)")
print("="*70)

# 5 - FEATURE IMPORTANCE (Permutation Importance)
print("\n" + "="*70)
print("CALCULATING FEATURE IMPORTANCE")
print("="*70)

# Calculate baseline score
baseline_pred = predict(X_test, y_test, parameters)
baseline_pred_inv = scaler_y.inverse_transform(baseline_pred.flatten().reshape(-1, 1)).flatten()
baseline_r2 = r2_score(y_actual_test_inv, baseline_pred_inv)

# Calculate permutation importance for each feature
feature_importance = []
n_permutations = 10  # Number of permutations for stability

for i, feature_name in enumerate(feature_columns):
    importance_scores = []

    for _ in range(n_permutations):
        # Create a copy of test data
        X_test_permuted = X_test.copy()

        # Shuffle the i-th feature (row in transposed data)
        np.random.shuffle(X_test_permuted[i, :])

        # Make predictions with permuted feature
        pred_permuted = predict(X_test_permuted, y_test, parameters)
        pred_permuted_inv = scaler_y.inverse_transform(pred_permuted.flatten().reshape(-1, 1)).flatten()

        # Calculate R² with permuted feature
        r2_permuted = r2_score(y_actual_test_inv, pred_permuted_inv)

        # Importance = drop in R² when feature is shuffled
        importance_scores.append(baseline_r2 - r2_permuted)

    avg_importance = np.mean(importance_scores)
    feature_importance.append({
        'Feature': feature_name,
        'Importance': avg_importance,
        'Std': np.std(importance_scores)
    })
    print(f"  {feature_name}: {avg_importance:.6f} (+/- {np.std(importance_scores):.6f})")

# Create importance dataframe and sort
importance_df = pd.DataFrame(feature_importance)
importance_df = importance_df.sort_values('Importance', ascending=True)

print("\n" + "="*70)
print("FEATURE IMPORTANCE RANKING (by R² drop)")
print("="*70)
for idx, row in importance_df.sort_values('Importance', ascending=False).iterrows():
    print(f"  {row['Feature']:<25}: {row['Importance']:.6f}")

# 6 - PLOT LAST 100 TEST DATA VS PREDICTIONS
print("\nGenerating prediction vs actual plot for last 100 test samples...")

num_samples = min(100, len(y_actual_test_inv))
last_100_actual = y_actual_test_inv[-num_samples:]
last_100_predicted = y_pred_test_inv[-num_samples:]

fig_pred, ax_pred = plt.subplots(figsize=(14, 6))

x_axis = np.arange(num_samples)

# Plot actual vs predicted
ax_pred.plot(x_axis, last_100_actual, 'b-', label='Actual Production', linewidth=1.5, alpha=0.8)
ax_pred.plot(x_axis, last_100_predicted, 'r--', label='Predicted Production', linewidth=1.5, alpha=0.8)

ax_pred.set_xlabel('Sample Index', fontsize=12)
ax_pred.set_ylabel('Production', fontsize=12)
ax_pred.set_title(f'Last {num_samples} Test Samples: Actual vs Predicted Production\n(Iterations: {NUM_ITERATIONS}, R²: {r2_test:.4f})', fontsize=14, fontweight='bold')
ax_pred.legend(fontsize=10, loc='upper right')
ax_pred.grid(True, alpha=0.3)

plt.tight_layout()
pred_plot_filename = f'DeepNN_Predictions_Last100_LR{LEARNING_RATE}.png'
plt.savefig(pred_plot_filename, dpi=300, bbox_inches='tight')
print(f"Prediction plot saved to: {pred_plot_filename}")
plt.show()

# 7 - PLOT FEATURE IMPORTANCE
print("\nGenerating feature importance plot...")

fig_imp, ax_imp = plt.subplots(figsize=(12, 8))

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))

bars = ax_imp.barh(importance_df['Feature'], importance_df['Importance'],
                   xerr=importance_df['Std'], color=colors, edgecolor='black', capsize=5)

ax_imp.set_xlabel('Importance (R² Drop when feature is permuted)', fontsize=12)
ax_imp.set_ylabel('Feature', fontsize=12)
ax_imp.set_title('Feature Importance for Production Prediction\n(Permutation Importance Method)', fontsize=14, fontweight='bold')
ax_imp.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for bar, importance in zip(bars, importance_df['Importance']):
    width = bar.get_width()
    ax_imp.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', fontsize=10)

plt.tight_layout()
importance_plot_filename = f'DeepNN_FeatureImportance_LR{LEARNING_RATE}.png'
plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
print(f"Feature importance plot saved to: {importance_plot_filename}")
plt.show()

# 8 - FINAL STATISTICS
print("\n" + "#"*70)
print("### FINAL STATISTICS")
print("#"*70)

print(f"\nTest R²: {r2_test:.6f}")
print(f"Test MAE: {mae_test:.4f}")
print(f"Test MSE: {mse_test:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")

print(f"\nTrain R²: {r2_train:.6f}")

print(f"\nComputation Time: {training_time:.2f}s ({training_time/60:.2f} min)")

# Most important features
print("\nTop 3 Most Important Features:")
top_features = importance_df.sort_values('Importance', ascending=False).head(3)
for idx, row in top_features.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.6f}")

print("\n" + "#"*70)
print("### TRAINING COMPLETE!")
print("#"*70)
