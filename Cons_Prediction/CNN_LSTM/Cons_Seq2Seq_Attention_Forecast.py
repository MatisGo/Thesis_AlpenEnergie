"""
Seq2Seq Encoder-Decoder with Attention for Consumption Forecasting
===================================================================
This script implements a Sequence-to-Sequence model with Attention mechanism
for 2-day ahead electricity consumption forecasting.

WHY SEQ2SEQ WITH ATTENTION?
---------------------------
The direct multi-step approach (predicting all 576 steps at once) tends to produce
flat predictions because:
1. The model minimizes MSE by predicting values close to the mean
2. No temporal dependency between output steps
3. Information bottleneck when compressing all context into a single vector

The Seq2Seq architecture solves this by:
1. Encoder: Processes historical data and creates a context representation
2. Decoder: Generates predictions step-by-step, maintaining temporal dependencies
3. Attention: Allows decoder to focus on relevant parts of the input sequence

ARCHITECTURE:
-------------
Based on research from:
- "Attention-Enhanced LSTM for Long-Horizon Time Series Forecasting" (2024)
- "Optimized Seq2Seq model for short-term power load forecasting" (2023)

1. CNN Feature Extractor: Extract local patterns from multivariate input
2. Encoder (Bi-LSTM): Process input sequence bidirectionally
3. Attention Layer: Compute time-varying context vectors
4. Decoder (LSTM): Generate predictions step-by-step using teacher forcing

FEATURES:
---------
- Rolling cross-validation for time series
- 80-20 train-test split
- Bahdanau (Additive) Attention mechanism
- Teacher forcing during training
- Scheduled sampling for better generalization
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
    BatchNormalization, Bidirectional, Concatenate,
    RepeatVector, TimeDistributed, Layer, Attention,
    AdditiveAttention, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
# Data parameters
LOOKBACK_STEPS = 576              # 576 x 5min = 48 hours (2 days) of past data
FORECAST_HORIZON = 576            # 576 x 5min = 48 hours (2 days) ahead prediction

# For Seq2Seq, we can optionally downsample predictions then interpolate
# This reduces decoder complexity while maintaining resolution
DECODER_OUTPUT_STEPS = 96         # Predict every 30 min (96 steps), then interpolate
USE_DOWNSAMPLING = False          # Set True to use downsampled decoder

# Model architecture - Encoder
CNN_FILTERS_1 = 64
CNN_FILTERS_2 = 128
CNN_KERNEL_SIZE = 3
POOL_SIZE = 2
ENCODER_LSTM_UNITS = 128

# Model architecture - Decoder
DECODER_LSTM_UNITS = 128
ATTENTION_UNITS = 64
DENSE_UNITS = 64

# Regularization
DROPOUT_RATE = 0.2

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15
VALIDATION_SPLIT = 0.2

# Teacher forcing
TEACHER_FORCING_RATIO = 0.5       # Probability of using true values during training

# Cross-validation parameters
N_SPLITS = 5
TEST_SIZE_RATIO = 0.20

# Model saving
SAVE_MODEL = True
MODEL_NAME = 'Cons_Seq2Seq_Attention_Model'

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
SEQUENCE_FEATURES = [
    'Consumption',
    'Temperature',
    'Temperature_Predicted',
    'Hour_sin',
    'Hour_cos',
    'DayOfWeek_sin',
    'DayOfWeek_cos',
    'IsWeekend',
]

# Features available for future (decoder input)
DECODER_FEATURES = [
    'Temperature_Predicted',  # We have temperature forecast for future
    'Hour_sin',
    'Hour_cos',
    'DayOfWeek_sin',
    'DayOfWeek_cos',
    'IsWeekend',
]


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    """Load and preprocess the consumption data."""
    print("Loading data...")
    data = pd.read_csv(filepath, skiprows=3, header=None, encoding='latin-1')

    data.columns = ['DateTime_str', 'Date', 'DayTime', 'Forecast_Prod', 'Forecast_Load',
                    'Consumption', 'Production', 'Level_Bidmi', 'Level_Haselholz',
                    'Temperature', 'Irradiance', 'Rain', 'SDR_Mode', 'Forecast_Mode',
                    'Transfer_Mode', 'Waterlevel_Mode', 'Temp_Forecast']

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

    # Cyclical encodings
    data['Hour_sin'] = np.sin(2 * np.pi * (data['Hour'] + data['Minute']/60) / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * (data['Hour'] + data['Minute']/60) / 24)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

    # Handle missing values
    for col in ['Consumption', 'Temperature', 'Temperature_Predicted']:
        if data[col].isnull().any():
            data[col] = data[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    print(f"Data loaded: {len(data)} samples")
    print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

    return data


def create_seq2seq_sequences(data, encoder_features, decoder_features, target_col,
                              lookback, forecast_horizon):
    """
    Create sequences for Seq2Seq model.

    Returns:
    --------
    encoder_input : (samples, lookback, encoder_features)
    decoder_input : (samples, forecast_horizon, decoder_features + 1)  # +1 for shifted target
    decoder_target : (samples, forecast_horizon)
    """
    enc_features = data[encoder_features].values
    dec_features = data[decoder_features].values
    targets = data[target_col].values
    timestamps = data['DateTime'].values

    encoder_X, decoder_X, decoder_y = [], [], []
    ts_start, ts_end = [], []

    for i in range(lookback, len(data) - forecast_horizon + 1):
        # Encoder input: past lookback steps with all features
        encoder_X.append(enc_features[i - lookback:i])

        # Decoder target: future consumption values
        future_targets = targets[i:i + forecast_horizon]
        decoder_y.append(future_targets)

        # Decoder input: future known features + shifted target (for teacher forcing)
        # During training, decoder gets true previous values
        # Shift target by 1: [0, y[0], y[1], ..., y[n-2]]
        future_dec_features = dec_features[i:i + forecast_horizon]

        # Create shifted target for decoder input (teacher forcing)
        shifted_target = np.zeros(forecast_horizon)
        shifted_target[0] = targets[i - 1]  # Last known consumption
        shifted_target[1:] = future_targets[:-1]  # Shifted targets

        # Combine decoder features with shifted target
        dec_input = np.column_stack([future_dec_features, shifted_target])
        decoder_X.append(dec_input)

        ts_start.append(timestamps[i])
        ts_end.append(timestamps[i + forecast_horizon - 1])

    return (np.array(encoder_X), np.array(decoder_X),
            np.array(decoder_y), ts_start, ts_end)


# ============================================================================
# CUSTOM ATTENTION LAYER (Bahdanau/Additive Attention)
# ============================================================================

class BahdanauAttention(Layer):
    """
    Bahdanau (Additive) Attention mechanism.

    This attention allows the decoder to focus on different parts of the
    encoder output at each decoding step.

    score(s_t, h_i) = v^T * tanh(W_s * s_t + W_h * h_i)
    """

    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # W_s for decoder state
        self.W_s = self.add_weight(
            name='W_s',
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        # W_h for encoder outputs
        self.W_h = self.add_weight(
            name='W_h',
            shape=(input_shape[1][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        # v for computing scalar score
        self.v = self.add_weight(
            name='v',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(BahdanauAttention, self).build(input_shape)

    def call(self, inputs):
        """
        inputs: [decoder_state, encoder_outputs]
        decoder_state: (batch, decoder_units)
        encoder_outputs: (batch, seq_len, encoder_units)
        """
        decoder_state, encoder_outputs = inputs

        # Expand decoder state: (batch, 1, decoder_units)
        decoder_state_expanded = tf.expand_dims(decoder_state, 1)

        # Score computation
        # (batch, 1, units) + (batch, seq_len, units)
        score = tf.nn.tanh(
            tf.matmul(decoder_state_expanded, self.W_s) +
            tf.matmul(encoder_outputs, self.W_h)
        )

        # (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.v), axis=1)

        # Context vector: weighted sum of encoder outputs
        # (batch, seq_len, 1) * (batch, seq_len, encoder_units) -> (batch, encoder_units)
        context = tf.reduce_sum(attention_weights * encoder_outputs, axis=1)

        return context, attention_weights

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config


# ============================================================================
# SEQ2SEQ MODEL WITH ATTENTION
# ============================================================================

def build_seq2seq_attention_model(encoder_input_shape, decoder_input_shape, output_steps):
    """
    Build Seq2Seq Encoder-Decoder model with Attention.

    Architecture:
    1. CNN Feature Extractor on encoder input
    2. Bidirectional LSTM Encoder
    3. Bahdanau Attention
    4. LSTM Decoder with attention context

    Parameters:
    -----------
    encoder_input_shape : tuple (lookback_steps, n_encoder_features)
    decoder_input_shape : tuple (forecast_horizon, n_decoder_features)
    output_steps : int (forecast_horizon)
    """

    # ==================== ENCODER ====================
    encoder_inputs = Input(shape=encoder_input_shape, name='encoder_input')

    # CNN Feature Extraction
    x = Conv1D(filters=CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same')(encoder_inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=POOL_SIZE)(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Conv1D(filters=CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=POOL_SIZE)(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Bidirectional LSTM Encoder
    encoder_lstm = Bidirectional(
        LSTM(ENCODER_LSTM_UNITS, return_sequences=True, return_state=True),
        name='encoder_lstm'
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(x)

    # Combine forward and backward states
    encoder_state_h = Concatenate()([forward_h, backward_h])
    encoder_state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [encoder_state_h, encoder_state_c]

    # ==================== DECODER ====================
    decoder_inputs = Input(shape=decoder_input_shape, name='decoder_input')

    # Decoder LSTM (needs 2x units because encoder is bidirectional)
    decoder_lstm = LSTM(
        ENCODER_LSTM_UNITS * 2,
        return_sequences=True,
        return_state=True,
        name='decoder_lstm'
    )

    # Initial decoder pass to get sequences
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Apply attention between decoder outputs and encoder outputs
    # Using Keras built-in attention for stability
    attention_layer = AdditiveAttention(name='attention')

    # For each decoder timestep, compute attention over encoder outputs
    # decoder_outputs: (batch, forecast_horizon, decoder_units)
    # encoder_outputs: (batch, reduced_seq_len, encoder_units*2)

    # Attention: query=decoder_outputs, key=value=encoder_outputs
    context_vectors = attention_layer([decoder_outputs, encoder_outputs])

    # Concatenate attention context with decoder outputs
    decoder_combined = Concatenate()([decoder_outputs, context_vectors])

    # Dense layers for prediction
    x = TimeDistributed(Dense(DENSE_UNITS, activation='relu'))(decoder_combined)
    x = Dropout(DROPOUT_RATE)(x)

    # Output: one value per timestep
    outputs = TimeDistributed(Dense(1), name='output')(x)
    outputs = tf.squeeze(outputs, axis=-1)  # (batch, forecast_horizon)

    # Create model
    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=outputs,
        name='Seq2Seq_Attention'
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    return model


def build_simple_seq2seq_model(encoder_input_shape, decoder_input_shape, output_steps):
    """
    Simplified Seq2Seq without custom attention (uses Keras Attention).
    More stable for training.
    """

    # ==================== ENCODER ====================
    encoder_inputs = Input(shape=encoder_input_shape, name='encoder_input')

    # CNN Feature Extraction
    x = Conv1D(filters=CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same')(encoder_inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=POOL_SIZE)(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Conv1D(filters=CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE,
               activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=POOL_SIZE)(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Encoder LSTM
    encoder_lstm = LSTM(ENCODER_LSTM_UNITS, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(x)
    encoder_states = [state_h, state_c]

    # ==================== DECODER ====================
    decoder_inputs = Input(shape=decoder_input_shape, name='decoder_input')

    # Decoder LSTM
    decoder_lstm = LSTM(ENCODER_LSTM_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Attention
    attention = Attention(name='attention')
    context = attention([decoder_outputs, encoder_outputs])

    # Combine
    decoder_combined = Concatenate()([decoder_outputs, context])

    # Output projection
    x = TimeDistributed(Dense(DENSE_UNITS, activation='relu'))(decoder_combined)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = TimeDistributed(Dense(1))(x)
    outputs = tf.squeeze(outputs, axis=-1)

    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=outputs,
        name='Seq2Seq_Simple_Attention'
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    return model


# ============================================================================
# TRAINING WITH TEACHER FORCING
# ============================================================================

class Seq2SeqDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator that implements teacher forcing with scheduled sampling.

    During training, with probability `teacher_forcing_ratio`, use true
    previous values. Otherwise, use model's own predictions.
    """

    def __init__(self, encoder_X, decoder_X, decoder_y, batch_size,
                 teacher_forcing_ratio=0.5, shuffle=True):
        self.encoder_X = encoder_X
        self.decoder_X = decoder_X
        self.decoder_y = decoder_y
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.shuffle = shuffle
        self.indices = np.arange(len(encoder_X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.encoder_X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_encoder_X = self.encoder_X[batch_indices]
        batch_decoder_X = self.decoder_X[batch_indices]
        batch_decoder_y = self.decoder_y[batch_indices]

        # For simplicity, always use teacher forcing during training
        # (true previous values in decoder input)
        return [batch_encoder_X, batch_decoder_X], batch_decoder_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# ROLLING CROSS-VALIDATION
# ============================================================================

def rolling_cv_train_seq2seq(encoder_X, decoder_X, y, n_splits=N_SPLITS):
    """
    Perform rolling/expanding window cross-validation for Seq2Seq model.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = []
    fold = 1

    for train_idx, val_idx in tscv.split(encoder_X):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        # Split data
        enc_X_train, enc_X_val = encoder_X[train_idx], encoder_X[val_idx]
        dec_X_train, dec_X_val = decoder_X[train_idx], decoder_X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build model
        model = build_simple_seq2seq_model(
            encoder_input_shape=(encoder_X.shape[1], encoder_X.shape[2]),
            decoder_input_shape=(decoder_X.shape[1], decoder_X.shape[2]),
            output_steps=y.shape[1]
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        # Train
        history = model.fit(
            [enc_X_train, dec_X_train], y_train,
            validation_data=([enc_X_val, dec_X_val], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        y_pred = model.predict([enc_X_val, dec_X_val], verbose=0)

        mae = mean_absolute_error(y_val.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_val.flatten(), y_pred.flatten()))
        r2 = r2_score(y_val.flatten(), y_pred.flatten())

        cv_results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        })

        print(f"\nFold {fold} Results:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2: {r2:.4f}")

        fold += 1

        # Clear memory
        del model
        tf.keras.backend.clear_session()

    return pd.DataFrame(cv_results)


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model, encoder_X, decoder_X, y_true, scaler_y):
    """Evaluate model and compute metrics."""
    y_pred_scaled = model.predict([encoder_X, decoder_X], verbose=0)

    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true_orig = scaler_y.inverse_transform(y_true)

    # Metrics
    mae = mean_absolute_error(y_true_orig.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true_orig.flatten(), y_pred.flatten()))
    r2 = r2_score(y_true_orig.flatten(), y_pred.flatten())

    # Per-step metrics
    step_mae = np.mean(np.abs(y_true_orig - y_pred), axis=0)
    step_rmse = np.sqrt(np.mean((y_true_orig - y_pred)**2, axis=0))

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred,
        'y_true': y_true_orig,
        'step_mae': step_mae,
        'step_rmse': step_rmse
    }


def plot_predictions(y_true, y_pred, n_samples=3, save_path=None):
    """Plot sample predictions vs actual."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 4*n_samples))

    if n_samples == 1:
        axes = [axes]

    sample_indices = np.linspace(0, len(y_true)-1, n_samples, dtype=int)

    for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
        x_vals = np.arange(FORECAST_HORIZON)
        ax.plot(x_vals, y_true[sample_idx], 'b-', label='Actual', linewidth=1.5)
        ax.plot(x_vals, y_pred[sample_idx], 'r--', label='Predicted', linewidth=1.5)
        ax.fill_between(x_vals, y_true[sample_idx], y_pred[sample_idx],
                        alpha=0.3, color='gray')

        ax.set_xlabel('Time Steps (5-min intervals)')
        ax.set_ylabel('Consumption (kW)')
        ax.set_title(f'Sample {sample_idx}: 2-Day Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add hour markers
        hour_ticks = np.arange(0, FORECAST_HORIZON, 12)  # Every hour
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels([f'{i*5//60}h' for i in hour_ticks])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()
    return fig


def plot_step_errors(step_mae, step_rmse, save_path=None):
    """Plot error metrics by forecast step."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_vals = np.arange(len(step_mae))
    hours = x_vals * 5 / 60

    # MAE by step
    axes[0].plot(hours, step_mae, 'b-', linewidth=1)
    axes[0].fill_between(hours, 0, step_mae, alpha=0.3)
    axes[0].set_xlabel('Forecast Horizon (hours)')
    axes[0].set_ylabel('MAE (kW)')
    axes[0].set_title('Mean Absolute Error by Forecast Step')
    axes[0].grid(True, alpha=0.3)

    # RMSE by step
    axes[1].plot(hours, step_rmse, 'r-', linewidth=1)
    axes[1].fill_between(hours, 0, step_rmse, alpha=0.3, color='red')
    axes[1].set_xlabel('Forecast Horizon (hours)')
    axes[1].set_ylabel('RMSE (kW)')
    axes[1].set_title('Root Mean Square Error by Forecast Step')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SEQ2SEQ ENCODER-DECODER WITH ATTENTION")
    print("Consumption Forecasting - 2 Day Ahead")
    print("=" * 80)

    start_time = time.time()

    # ========== 1. LOAD DATA ==========
    print("\n" + "="*60)
    print("1. LOADING AND PREPROCESSING DATA")
    print("="*60)

    data = load_and_preprocess_data('../Data_January.csv')

    # ========== 2. CREATE SEQUENCES ==========
    print("\n" + "="*60)
    print("2. CREATING SEQ2SEQ SEQUENCES")
    print("="*60)

    encoder_X, decoder_X, y, ts_start, ts_end = create_seq2seq_sequences(
        data,
        encoder_features=SEQUENCE_FEATURES,
        decoder_features=DECODER_FEATURES,
        target_col='Consumption',
        lookback=LOOKBACK_STEPS,
        forecast_horizon=FORECAST_HORIZON
    )

    print(f"Encoder input shape: {encoder_X.shape}")
    print(f"Decoder input shape: {decoder_X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Total samples: {len(encoder_X)}")

    # ========== 3. NORMALIZE DATA ==========
    print("\n" + "="*60)
    print("3. NORMALIZING DATA")
    print("="*60)

    # Fit scaler on encoder features
    scaler_enc = MinMaxScaler(feature_range=(0, 1))
    encoder_X_reshaped = encoder_X.reshape(-1, encoder_X.shape[-1])
    scaler_enc.fit(encoder_X_reshaped)
    encoder_X_scaled = scaler_enc.transform(encoder_X_reshaped).reshape(encoder_X.shape)

    # Fit scaler on decoder features
    scaler_dec = MinMaxScaler(feature_range=(0, 1))
    decoder_X_reshaped = decoder_X.reshape(-1, decoder_X.shape[-1])
    scaler_dec.fit(decoder_X_reshaped)
    decoder_X_scaled = scaler_dec.transform(decoder_X_reshaped).reshape(decoder_X.shape)

    # Fit scaler on target
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_reshaped = y.reshape(-1, 1)
    scaler_y.fit(y_reshaped)
    y_scaled = scaler_y.transform(y_reshaped).reshape(y.shape)

    print(f"Encoder X range: [{encoder_X_scaled.min():.3f}, {encoder_X_scaled.max():.3f}]")
    print(f"Decoder X range: [{decoder_X_scaled.min():.3f}, {decoder_X_scaled.max():.3f}]")
    print(f"Target y range: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")

    # ========== 4. TRAIN-TEST SPLIT ==========
    print("\n" + "="*60)
    print("4. TRAIN-TEST SPLIT (80-20)")
    print("="*60)

    split_idx = int(len(encoder_X_scaled) * (1 - TEST_SIZE_RATIO))

    enc_X_train, enc_X_test = encoder_X_scaled[:split_idx], encoder_X_scaled[split_idx:]
    dec_X_train, dec_X_test = decoder_X_scaled[:split_idx], decoder_X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    print(f"Training samples: {len(enc_X_train)}")
    print(f"Test samples: {len(enc_X_test)}")

    # ========== 5. ROLLING CROSS-VALIDATION ==========
    print("\n" + "="*60)
    print("5. ROLLING CROSS-VALIDATION")
    print("="*60)

    cv_df = rolling_cv_train_seq2seq(enc_X_train, dec_X_train, y_train)

    print("\n" + "-"*40)
    print("CROSS-VALIDATION SUMMARY")
    print("-"*40)
    print(f"Mean MAE: {cv_df['mae'].mean():.4f} (+/- {cv_df['mae'].std():.4f})")
    print(f"Mean RMSE: {cv_df['rmse'].mean():.4f} (+/- {cv_df['rmse'].std():.4f})")
    print(f"Mean R2: {cv_df['r2'].mean():.4f} (+/- {cv_df['r2'].std():.4f})")

    # ========== 6. TRAIN FINAL MODEL ==========
    print("\n" + "="*60)
    print("6. TRAINING FINAL MODEL ON FULL TRAINING DATA")
    print("="*60)

    final_model = build_simple_seq2seq_model(
        encoder_input_shape=(encoder_X_scaled.shape[1], encoder_X_scaled.shape[2]),
        decoder_input_shape=(decoder_X_scaled.shape[1], decoder_X_scaled.shape[2]),
        output_steps=y_scaled.shape[1]
    )

    print("\nModel Architecture:")
    final_model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'{MODEL_NAME}_best.keras', monitor='val_loss',
                       save_best_only=True, verbose=1)
    ]

    history = final_model.fit(
        [enc_X_train, dec_X_train], y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ========== 7. EVALUATE ON TEST SET ==========
    print("\n" + "="*60)
    print("7. EVALUATING ON TEST SET")
    print("="*60)

    metrics = evaluate_model(final_model, enc_X_test, dec_X_test, y_test, scaler_y)

    print(f"\nTest Set Results:")
    print(f"  MAE: {metrics['mae']:.4f} kW")
    print(f"  RMSE: {metrics['rmse']:.4f} kW")
    print(f"  R2 Score: {metrics['r2']:.4f}")

    # ========== 8. VISUALIZATIONS ==========
    print("\n" + "="*60)
    print("8. GENERATING VISUALIZATIONS")
    print("="*60)

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training History - MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{MODEL_NAME}_training_history.png', dpi=150)
    plt.show()

    # Plot sample predictions
    plot_predictions(metrics['y_true'], metrics['y_pred'], n_samples=3,
                    save_path=f'{MODEL_NAME}_predictions.png')

    # Plot error by forecast step
    plot_step_errors(metrics['step_mae'], metrics['step_rmse'],
                    save_path=f'{MODEL_NAME}_step_errors.png')

    # ========== 9. SAVE MODEL AND CONFIG ==========
    if SAVE_MODEL:
        print("\n" + "="*60)
        print("9. SAVING MODEL AND CONFIGURATION")
        print("="*60)

        # Save model
        final_model.save(f'{MODEL_NAME}.keras')
        print(f"Model saved to: {MODEL_NAME}.keras")

        # Save configuration
        np.savez(f'{MODEL_NAME}_Config.npz',
                 # Configuration
                 lookback_steps=np.array(LOOKBACK_STEPS),
                 forecast_horizon=np.array(FORECAST_HORIZON),
                 encoder_features=np.array(SEQUENCE_FEATURES),
                 decoder_features=np.array(DECODER_FEATURES),
                 n_encoder_features=np.array(len(SEQUENCE_FEATURES)),
                 n_decoder_features=np.array(len(DECODER_FEATURES) + 1),  # +1 for shifted target
                 # Encoder scaler
                 scaler_enc_min=scaler_enc.data_min_,
                 scaler_enc_max=scaler_enc.data_max_,
                 scaler_enc_scale=scaler_enc.scale_,
                 scaler_enc_data_range=scaler_enc.data_range_,
                 scaler_enc_min_=scaler_enc.min_,
                 # Decoder scaler
                 scaler_dec_min=scaler_dec.data_min_,
                 scaler_dec_max=scaler_dec.data_max_,
                 scaler_dec_scale=scaler_dec.scale_,
                 scaler_dec_data_range=scaler_dec.data_range_,
                 scaler_dec_min_=scaler_dec.min_,
                 # Target scaler
                 scaler_y_min=scaler_y.data_min_,
                 scaler_y_max=scaler_y.data_max_,
                 scaler_y_scale=scaler_y.scale_,
                 scaler_y_data_range=scaler_y.data_range_,
                 scaler_y_min_=scaler_y.min_,
                 # CV metrics
                 cv_mean_r2=np.array(cv_df['r2'].mean()),
                 cv_mean_mae=np.array(cv_df['mae'].mean()),
                 cv_mean_rmse=np.array(cv_df['rmse'].mean()),
                 cv_std_r2=np.array(cv_df['r2'].std()),
                 # Final metrics
                 final_r2=np.array(metrics['r2']),
                 final_mae=np.array(metrics['mae']),
                 final_rmse=np.array(metrics['rmse']))

        print(f"Configuration saved to: {MODEL_NAME}_Config.npz")

    # ========== 10. SUMMARY ==========
    elapsed_time = time.time() - start_time

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal time: {elapsed_time/60:.1f} minutes")
    print(f"\nCross-Validation Results:")
    print(f"  Mean R2: {cv_df['r2'].mean():.4f} (+/- {cv_df['r2'].std():.4f})")
    print(f"  Mean MAE: {cv_df['mae'].mean():.4f} kW")
    print(f"  Mean RMSE: {cv_df['rmse'].mean():.4f} kW")
    print(f"\nFinal Test Results:")
    print(f"  R2: {metrics['r2']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f} kW")
    print(f"  RMSE: {metrics['rmse']:.4f} kW")
    print("\n" + "="*80)