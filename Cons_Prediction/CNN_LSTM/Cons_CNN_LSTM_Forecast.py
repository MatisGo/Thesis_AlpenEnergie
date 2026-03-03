"""
Hybrid CNN-LSTM Consumption Forecasting Model
==============================================
This script implements a hybrid CNN-LSTM model for 2-day ahead electricity
consumption forecasting based on the architecture from:

1. Chung & Jang (2022) - "Accurate prediction of electricity consumption using
   a hybrid CNN-LSTM model based on multivariable data" (PLoS ONE)

2. Khan et al. (2020) - "Towards Efficient Electricity Forecasting in Residential
   and Commercial Buildings: A Novel Hybrid CNN with a LSTM-AE based Framework" (Sensors)

KEY ARCHITECTURE:
-----------------
1. CNN layers (1D): Extract spatial features from multivariate time series
   - Conv1D with ReLU activation
   - MaxPooling for dimensionality reduction
   - Dropout for regularization

2. LSTM layers: Capture temporal dependencies
   - LSTM cells process the CNN features sequentially
   - Return sequences for multi-step output

3. Dense layers: Generate multi-step predictions
   - Outputs 576 values (2 days at 5-min resolution)

FEATURES:
---------
- Rolling cross-validation for time series
- 80-20 train-test split
- Multi-step direct forecasting (predict all 576 steps at once)
- Support for future prediction with only past data + temperature forecast
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
                                      Flatten, Input, BatchNormalization, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================================
# HYPERPARAMETERS - EASY TO CHANGE
# ============================================================================
# Data parameters
LOOKBACK_STEPS = 576              # 576 x 5min = 48 hours (2 days) of past data
FORECAST_HORIZON = 576            # 576 x 5min = 48 hours (2 days) ahead prediction

# Model architecture
CNN_FILTERS_1 = 64                # Number of filters in first Conv1D layer
CNN_FILTERS_2 = 128               # Number of filters in second Conv1D layer
CNN_KERNEL_SIZE = 3               # Kernel size for Conv1D
POOL_SIZE = 2                     # MaxPooling size
LSTM_UNITS_1 = 128                # Units in first LSTM layer
LSTM_UNITS_2 = 64                 # Units in second LSTM layer
DENSE_UNITS = 128                 # Units in dense layer before output
DROPOUT_RATE = 0.2                # Dropout rate for regularization

# Training parameters
LEARNING_RATE = 0.001             # Learning rate for Adam optimizer
EPOCHS = 100                      # Maximum training epochs
BATCH_SIZE = 32                   # Batch size for training
PATIENCE = 15                     # Early stopping patience
VALIDATION_SPLIT = 0.2            # Validation split for training

# Cross-validation parameters
N_SPLITS = 5                      # Number of CV folds
TEST_SIZE_RATIO = 0.20            # 80-20 split (20% for test)

# Model saving
SAVE_MODEL = True
MODEL_NAME = 'Cons_CNN_LSTM_Model'

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# Features used in the model
SEQUENCE_FEATURES = [
    'Consumption',           # Past consumption values (main feature)
    'Temperature',           # Actual temperature
    'Temperature_Predicted', # Forecasted temperature (important for future prediction)
    'Hour_sin',             # Cyclical hour encoding (sin)
    'Hour_cos',             # Cyclical hour encoding (cos)
    'DayOfWeek_sin',        # Cyclical day encoding (sin)
    'DayOfWeek_cos',        # Cyclical day encoding (cos)
    'IsWeekend',            # Binary weekend indicator
]

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the consumption data.

    Returns preprocessed DataFrame with all necessary features.
    """
    print("Loading data...")
    data = pd.read_csv(filepath, skiprows=3, header=None, encoding='latin-1')

    # Assign column names
    data.columns = ['DateTime_str', 'Date', 'DayTime', 'Forecast_Prod', 'Forecast_Load',
                    'Consumption', 'Production', 'Level_Bidmi', 'Level_Haselholz',
                    'Temperature', 'Irradiance', 'Rain', 'SDR_Mode', 'Forecast_Mode',
                    'Transfer_Mode', 'Waterlevel_Mode', 'Temp_Forecast']

    # Parse DateTime
    data['DateTime'] = pd.to_datetime(data['DateTime_str'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    data = data.dropna(subset=['DateTime'])
    data = data.sort_values('DateTime').reset_index(drop=True)

    # Handle Temp_Forecast
    data['Temp_Forecast'] = pd.to_numeric(data['Temp_Forecast'], errors='coerce')
    data['Temp_Forecast'] = data['Temp_Forecast'].fillna(data['Temperature'])
    data.rename(columns={'Temp_Forecast': 'Temperature_Predicted'}, inplace=True)

    # Add date/time features
    data['Date'] = data['DateTime'].dt.date
    data['Hour'] = data['DateTime'].dt.hour
    data['Minute'] = data['DateTime'].dt.minute
    data['DayOfWeek'] = data['DateTime'].dt.dayofweek
    data['Month'] = data['DateTime'].dt.month
    data['DayOfYear'] = data['DateTime'].dt.dayofyear
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

    # Cyclical encodings for time features
    data['Hour_sin'] = np.sin(2 * np.pi * (data['Hour'] + data['Minute']/60) / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * (data['Hour'] + data['Minute']/60) / 24)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

    # Handle missing values through interpolation
    for col in ['Consumption', 'Temperature', 'Temperature_Predicted']:
        if data[col].isnull().any():
            data[col] = data[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    print(f"Data loaded: {len(data)} samples")
    print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

    return data


def create_sequences_multistep(data, feature_cols, target_col, lookback, forecast_horizon):
    """
    Create sequences for multi-step forecasting.

    Input shape: (samples, lookback, features)
    Output shape: (samples, forecast_horizon)

    Each sample contains:
    - X: lookback timesteps of all features
    - y: next forecast_horizon values of target
    """
    features = data[feature_cols].values
    targets = data[target_col].values
    timestamps = data['DateTime'].values

    X, y, ts_start, ts_end = [], [], [], []

    for i in range(lookback, len(data) - forecast_horizon + 1):
        # Input sequence: past lookback steps
        X.append(features[i - lookback:i])

        # Target: next forecast_horizon steps
        y.append(targets[i:i + forecast_horizon])

        # Timestamps for reference
        ts_start.append(timestamps[i])
        ts_end.append(timestamps[i + forecast_horizon - 1])

    return np.array(X), np.array(y), ts_start, ts_end


# ============================================================================
# CNN-LSTM MODEL ARCHITECTURE
# ============================================================================

def build_cnn_lstm_model(input_shape, output_size):
    """
    Build the hybrid CNN-LSTM model.

    Architecture based on papers:
    - Chung & Jang (2022): CNN for feature extraction + LSTM for temporal modeling
    - Khan et al. (2020): CNN with LSTM-AE framework

    Parameters:
    -----------
    input_shape : tuple
        Shape of input (timesteps, features)
    output_size : int
        Number of steps to predict (forecast_horizon)

    Returns:
    --------
    model : keras.Model
        Compiled CNN-LSTM model
    """
    model = Sequential([
        # ========== CNN Feature Extraction ==========
        # First Conv1D block
        Conv1D(filters=CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=POOL_SIZE),
        Dropout(DROPOUT_RATE),

        # Second Conv1D block
        Conv1D(filters=CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=POOL_SIZE),
        Dropout(DROPOUT_RATE),

        # ========== LSTM Temporal Modeling ==========
        # First LSTM layer (return sequences for stacking)
        LSTM(units=LSTM_UNITS_1, return_sequences=True),
        Dropout(DROPOUT_RATE),

        # Second LSTM layer
        LSTM(units=LSTM_UNITS_2, return_sequences=False),
        Dropout(DROPOUT_RATE),

        # ========== Dense Output Layers ==========
        Dense(units=DENSE_UNITS, activation='relu'),
        Dropout(DROPOUT_RATE),

        # Output layer: predict all forecast_horizon steps
        Dense(units=output_size)
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    return model


def build_cnn_bilstm_model(input_shape, output_size):
    """
    Alternative architecture using Bidirectional LSTM.

    Bidirectional LSTM processes sequences both forward and backward,
    which can capture more complex temporal patterns.
    """
    model = Sequential([
        # CNN Feature Extraction
        Conv1D(filters=CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=POOL_SIZE),
        Dropout(DROPOUT_RATE),

        Conv1D(filters=CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=POOL_SIZE),
        Dropout(DROPOUT_RATE),

        # Bidirectional LSTM
        Bidirectional(LSTM(units=LSTM_UNITS_1, return_sequences=True)),
        Dropout(DROPOUT_RATE),

        Bidirectional(LSTM(units=LSTM_UNITS_2, return_sequences=False)),
        Dropout(DROPOUT_RATE),

        # Dense output
        Dense(units=DENSE_UNITS, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(units=output_size)
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    return model


# ============================================================================
# ROLLING CROSS-VALIDATION
# ============================================================================

def rolling_cross_validation(X, y, timestamps_start, n_splits=5, test_ratio=0.2):
    """
    Perform rolling (expanding window) cross-validation for time series.

    Unlike standard k-fold CV, this respects temporal order:
    - Training set always comes before test set
    - Training window expands with each fold

    Parameters:
    -----------
    X : np.array
        Input features (samples, timesteps, features)
    y : np.array
        Targets (samples, forecast_horizon)
    timestamps_start : list
        Start timestamps for each sample
    n_splits : int
        Number of CV folds
    test_ratio : float
        Proportion of data for testing (0.2 = 80-20 split)

    Returns:
    --------
    cv_results : list of dicts
        Results for each fold
    """
    n_samples = len(X)
    test_size = int(n_samples * test_ratio / n_splits)

    # Calculate fold boundaries
    # Each fold uses more training data and a fixed test size
    fold_results = []

    # Use TimeSeriesSplit for proper temporal CV
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*70}")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Normalize features
        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_test = X_test.shape[0]

        # Reshape for scaler
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)

        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(n_samples_train, n_timesteps, n_features)
        X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(n_samples_test, n_timesteps, n_features)

        # Normalize targets
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        # Build and train model
        model = build_cnn_lstm_model(
            input_shape=(n_timesteps, n_features),
            output_size=y_train.shape[1]
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        start_time = time.time()
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
        train_time = time.time() - start_time

        # Predict
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # Calculate metrics
        mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
        mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
        mape = np.mean(np.abs((y_test.flatten() - y_pred.flatten()) / (y_test.flatten() + 1e-8))) * 100

        fold_result = {
            'fold': fold_idx + 1,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'epochs': len(history.history['loss']),
            'train_time': train_time
        }

        fold_results.append(fold_result)

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.2f} kW")
        print(f"  RMSE: {rmse:.2f} kW")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Time: {train_time:.2f}s")

        # Clean up to free memory
        del model
        tf.keras.backend.clear_session()

    return fold_results


# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================

def train_final_model(X, y, test_ratio=0.2):
    """
    Train the final model on the full dataset with 80-20 split.

    Parameters:
    -----------
    X : np.array
        Input features
    y : np.array
        Targets
    test_ratio : float
        Proportion for test set

    Returns:
    --------
    model : keras.Model
        Trained model
    scalers : tuple
        (scaler_X, scaler_y) for inference
    metrics : dict
        Performance metrics
    history : History
        Training history
    """
    # Split data (respecting temporal order)
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nFinal Model Training:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Normalize
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples_train, n_timesteps, n_features)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(n_samples_test, n_timesteps, n_features)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Build model
    model = build_cnn_lstm_model(
        input_shape=(n_timesteps, n_features),
        output_size=y_train.shape[1]
    )

    print("\nModel Architecture:")
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'{MODEL_NAME}.keras', monitor='val_loss', save_best_only=True)
    ]

    # Train
    start_time = time.time()
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time

    # Predict on test set
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test.flatten(), y_pred.flatten()),
        'mse': mean_squared_error(y_test.flatten(), y_pred.flatten()),
        'rmse': np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten())),
        'r2': r2_score(y_test.flatten(), y_pred.flatten()),
        'mape': np.mean(np.abs((y_test.flatten() - y_pred.flatten()) / (y_test.flatten() + 1e-8))) * 100,
        'train_time': train_time,
        'epochs': len(history.history['loss'])
    }

    return model, (scaler_X, scaler_y), metrics, history, (X_test, y_test, y_pred)


# ============================================================================
# FUTURE PREDICTION MODE
# ============================================================================

def predict_future(model, scaler_X, scaler_y, recent_data, feature_cols, temp_forecast=None):
    """
    Predict future consumption using only past data and temperature forecast.

    This function is used for real forecasting when we don't have actual
    future consumption values.

    Parameters:
    -----------
    model : keras.Model
        Trained CNN-LSTM model
    scaler_X : MinMaxScaler
        Fitted scaler for features
    scaler_y : MinMaxScaler
        Fitted scaler for targets
    recent_data : pd.DataFrame
        Recent data (at least LOOKBACK_STEPS rows)
    feature_cols : list
        Feature column names
    temp_forecast : np.array or None
        Temperature forecast for the prediction horizon (FORECAST_HORIZON values)
        If None, uses the last known Temperature_Predicted values

    Returns:
    --------
    predictions : np.array
        Predicted consumption values (FORECAST_HORIZON values)
    timestamps : list
        Timestamps for predictions
    """
    if len(recent_data) < LOOKBACK_STEPS:
        raise ValueError(f"Need at least {LOOKBACK_STEPS} samples of recent data")

    # Get the last LOOKBACK_STEPS samples
    input_data = recent_data.tail(LOOKBACK_STEPS).copy()

    # If temperature forecast is provided, we could potentially use it
    # For now, we use the existing Temperature_Predicted values

    # Extract features
    X = input_data[feature_cols].values

    # Normalize
    n_timesteps, n_features = X.shape
    X_scaled = scaler_X.transform(X.reshape(-1, n_features)).reshape(1, n_timesteps, n_features)

    # Predict
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # Generate future timestamps
    last_timestamp = recent_data['DateTime'].iloc[-1]
    future_timestamps = [last_timestamp + pd.Timedelta(minutes=5 * (i + 1)) for i in range(FORECAST_HORIZON)]

    return y_pred, future_timestamps


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(y_test, y_pred, cv_results, history, output_dir='./'):
    """
    Create comprehensive visualization of results.
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    # Plot 1: CV Results - R² across folds
    ax1 = axes[0, 0]
    folds = [r['fold'] for r in cv_results]
    r2_scores = [r['r2'] for r in cv_results]
    ax1.bar(folds, r2_scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=np.mean(r2_scores), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(r2_scores):.4f}')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Cross-Validation R² Scores', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: CV Results - MAE and RMSE
    ax2 = axes[0, 1]
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
    ax2.set_xticklabels([f'Fold {r["fold"]}' for r in cv_results])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sample prediction vs actual (first 2 days from test set)
    ax3 = axes[1, 0]
    sample_idx = 0
    actual = y_test[sample_idx]
    predicted = y_pred[sample_idx]
    x_vals = np.arange(len(actual))
    ax3.plot(x_vals, actual, 'b-', label='Actual', linewidth=1.5)
    ax3.plot(x_vals, predicted, 'r--', label='Predicted', linewidth=1.5)
    ax3.fill_between(x_vals, actual, predicted, alpha=0.3, color='gray')
    ax3.set_xlabel('Time Step (5-min intervals)', fontsize=12)
    ax3.set_ylabel('Consumption (kW)', fontsize=12)
    ax3.set_title('Sample 2-Day Forecast vs Actual', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Prediction error distribution
    ax4 = axes[1, 1]
    errors = y_pred.flatten() - y_test.flatten()
    ax4.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2,
                label=f'Mean Error: {np.mean(errors):.2f} kW')
    ax4.set_xlabel('Prediction Error (kW)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Training history
    ax5 = axes[2, 0]
    ax5.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax5.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss (MSE)', fontsize=12)
    ax5.set_title('Training History', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Actual vs Predicted scatter
    ax6 = axes[2, 1]
    ax6.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.3, s=5, c='steelblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax6.set_xlabel('Actual Consumption (kW)', fontsize=12)
    ax6.set_ylabel('Predicted Consumption (kW)', fontsize=12)
    ax6.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'CNN_LSTM_Results.png'), dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HYBRID CNN-LSTM CONSUMPTION FORECASTING")
    print("2-Day Ahead Multi-Step Prediction")
    print("=" * 80)

    # 1. Load and preprocess data
    print("\n" + "=" * 80)
    print("1. LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    data = load_and_preprocess_data('../Data_January.csv')

    # 2. Create sequences
    print("\n" + "=" * 80)
    print("2. CREATING SEQUENCES")
    print("=" * 80)
    print(f"Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
    print(f"Forecast horizon: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * 5 / 60:.1f} hours)")

    X, y, ts_start, ts_end = create_sequences_multistep(
        data, SEQUENCE_FEATURES, 'Consumption', LOOKBACK_STEPS, FORECAST_HORIZON
    )

    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # 3. Rolling Cross-Validation
    print("\n" + "=" * 80)
    print("3. ROLLING CROSS-VALIDATION")
    print("=" * 80)

    cv_results = rolling_cross_validation(X, y, ts_start, n_splits=N_SPLITS, test_ratio=TEST_SIZE_RATIO)

    # Print CV Summary
    print("\n" + "-" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("-" * 80)
    cv_df = pd.DataFrame(cv_results)
    print(f"\n{'Metric':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 60)
    for metric in ['r2', 'mae', 'rmse', 'mape']:
        print(f"{metric.upper():<10} {cv_df[metric].mean():>12.4f} {cv_df[metric].std():>12.4f} "
              f"{cv_df[metric].min():>12.4f} {cv_df[metric].max():>12.4f}")

    # Save CV results
    cv_df.to_csv('CNN_LSTM_CV_Results.csv', index=False)
    print(f"\nCV results saved to: CNN_LSTM_CV_Results.csv")

    # 4. Train Final Model
    print("\n" + "=" * 80)
    print("4. TRAINING FINAL MODEL")
    print("=" * 80)

    model, scalers, metrics, history, test_data = train_final_model(X, y, test_ratio=TEST_SIZE_RATIO)
    X_test, y_test, y_pred = test_data

    print("\n" + "-" * 80)
    print("FINAL MODEL PERFORMANCE")
    print("-" * 80)
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAE:  {metrics['mae']:.2f} kW")
    print(f"RMSE: {metrics['rmse']:.2f} kW")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Training time: {metrics['train_time']:.2f}s")
    print(f"Epochs: {metrics['epochs']}")

    # 5. Visualization
    print("\n" + "=" * 80)
    print("5. GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_results(y_test, y_pred, cv_results, history)

    # 6. Save Model
    if SAVE_MODEL:
        print("\n" + "=" * 80)
        print("6. SAVING MODEL AND CONFIGURATION")
        print("=" * 80)

        # Save scalers and configuration
        scaler_X, scaler_y = scalers
        np.savez(f'{MODEL_NAME}_Config.npz',
                 # Configuration
                 lookback_steps=np.array(LOOKBACK_STEPS),
                 forecast_horizon=np.array(FORECAST_HORIZON),
                 feature_columns=np.array(SEQUENCE_FEATURES),
                 n_features=np.array(len(SEQUENCE_FEATURES)),
                 # Scaler X (all attributes needed for transform)
                 scaler_X_min=scaler_X.data_min_,
                 scaler_X_max=scaler_X.data_max_,
                 scaler_X_scale=scaler_X.scale_,
                 scaler_X_data_range=scaler_X.data_range_,
                 scaler_X_min_=scaler_X.min_,  # Required for transform
                 # Scaler y (all attributes needed for transform/inverse_transform)
                 scaler_y_min=scaler_y.data_min_,
                 scaler_y_max=scaler_y.data_max_,
                 scaler_y_scale=scaler_y.scale_,
                 scaler_y_data_range=scaler_y.data_range_,
                 scaler_y_min_=scaler_y.min_,  # Required for transform
                 # CV metrics
                 cv_mean_r2=np.array(cv_df['r2'].mean()),
                 cv_mean_mae=np.array(cv_df['mae'].mean()),
                 cv_mean_rmse=np.array(cv_df['rmse'].mean()),
                 cv_std_r2=np.array(cv_df['r2'].std()),
                 # Final metrics
                 final_r2=np.array(metrics['r2']),
                 final_mae=np.array(metrics['mae']),
                 final_rmse=np.array(metrics['rmse']),
                 final_mape=np.array(metrics['mape'])
        )

        print(f"Model saved to: {MODEL_NAME}.keras")
        print(f"Configuration saved to: {MODEL_NAME}_Config.npz")

    # 7. Test Future Prediction Mode
    print("\n" + "=" * 80)
    print("7. TESTING FUTURE PREDICTION MODE")
    print("=" * 80)

    # Get recent data for prediction
    recent_data = data.tail(LOOKBACK_STEPS + 100).copy()

    # Predict future
    future_pred, future_timestamps = predict_future(
        model, scaler_X, scaler_y, recent_data, SEQUENCE_FEATURES
    )

    print(f"Future prediction generated for {len(future_pred)} time steps")
    print(f"Prediction range: {future_timestamps[0]} to {future_timestamps[-1]}")
    print(f"Predicted mean consumption: {future_pred.mean():.2f} kW")
    print(f"Predicted min: {future_pred.min():.2f} kW")
    print(f"Predicted max: {future_pred.max():.2f} kW")

    # Save future predictions
    future_df = pd.DataFrame({
        'DateTime': future_timestamps,
        'Predicted_Consumption_kW': future_pred
    })
    future_df.to_csv('Future_Predictions_2Days.csv', index=False)
    print(f"\nFuture predictions saved to: Future_Predictions_2Days.csv")

    # Final Summary
    print("\n" + "#" * 80)
    print("### FINAL SUMMARY - CNN-LSTM CONSUMPTION FORECASTING")
    print("#" * 80)

    print(f"\n--- Model Configuration ---")
    print(f"Architecture: CNN-LSTM Hybrid")
    print(f"CNN Filters: {CNN_FILTERS_1}, {CNN_FILTERS_2}")
    print(f"LSTM Units: {LSTM_UNITS_1}, {LSTM_UNITS_2}")
    print(f"Lookback: {LOOKBACK_STEPS} steps ({LOOKBACK_STEPS * 5 / 60:.1f} hours)")
    print(f"Forecast Horizon: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * 5 / 60:.1f} hours)")

    print(f"\n--- Cross-Validation Performance ({N_SPLITS} folds) ---")
    print(f"Mean R²:   {cv_df['r2'].mean():.4f} (±{cv_df['r2'].std():.4f})")
    print(f"Mean MAE:  {cv_df['mae'].mean():.2f} kW (±{cv_df['mae'].std():.2f})")
    print(f"Mean RMSE: {cv_df['rmse'].mean():.2f} kW (±{cv_df['rmse'].std():.2f})")
    print(f"Mean MAPE: {cv_df['mape'].mean():.2f}% (±{cv_df['mape'].std():.2f})")

    print(f"\n--- Final Model Performance (80-20 split) ---")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAE:  {metrics['mae']:.2f} kW")
    print(f"RMSE: {metrics['rmse']:.2f} kW")
    print(f"MAPE: {metrics['mape']:.2f}%")

    print("\n" + "#" * 80)
    print("### CNN-LSTM FORECASTING COMPLETE!")
    print("#" * 80)
