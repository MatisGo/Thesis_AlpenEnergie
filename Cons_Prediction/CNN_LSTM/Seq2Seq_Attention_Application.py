"""
Seq2Seq Attention Model Application Script
==========================================
This script loads a trained Seq2Seq Encoder-Decoder with Attention model
and performs 2-day ahead consumption forecasting.

USE CASES:
----------
1. Real-time forecasting: Predict next 2 days based on latest data
2. Batch forecasting: Generate predictions for multiple starting points

REQUIREMENTS:
-------------
- Trained model: Cons_Seq2Seq_Attention_Model.keras
- Configuration: Cons_Seq2Seq_Attention_Model_Config.npz
- Input data: Recent consumption and temperature data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


class Seq2SeqForecaster:
    """
    Class for loading and using a trained Seq2Seq Attention model for consumption forecasting.
    """

    def __init__(self, model_path='Cons_Seq2Seq_Attention_Model.keras',
                 config_path='Cons_Seq2Seq_Attention_Model_Config.npz'):
        """
        Initialize the forecaster by loading the model and configuration.

        Parameters:
        -----------
        model_path : str
            Path to the trained Keras model
        config_path : str
            Path to the configuration file with scalers and parameters
        """
        self.model_path = model_path
        self.config_path = config_path

        self._load_model()
        self._load_config()

        print(f"Seq2Seq Attention Forecaster initialized")
        print(f"  Lookback: {self.lookback_steps} steps ({self.lookback_steps * 5 / 60:.1f} hours)")
        print(f"  Forecast horizon: {self.forecast_horizon} steps ({self.forecast_horizon * 5 / 60:.1f} hours)")

    def _load_model(self):
        """Load the trained Keras model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = load_model(self.model_path)
        print(f"Model loaded from: {self.model_path}")

    def _load_config(self):
        """Load the configuration and scalers."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        config = np.load(self.config_path, allow_pickle=True)

        # Model parameters
        self.lookback_steps = int(config['lookback_steps'])
        self.forecast_horizon = int(config['forecast_horizon'])
        self.encoder_features = list(config['encoder_features'])
        self.decoder_features = list(config['decoder_features'])
        self.n_encoder_features = int(config['n_encoder_features'])
        self.n_decoder_features = int(config['n_decoder_features'])

        # Reconstruct encoder scaler
        self.scaler_enc = MinMaxScaler(feature_range=(0, 1))
        self.scaler_enc.data_min_ = config['scaler_enc_min']
        self.scaler_enc.data_max_ = config['scaler_enc_max']
        self.scaler_enc.scale_ = config['scaler_enc_scale']
        self.scaler_enc.data_range_ = config['scaler_enc_data_range']
        self.scaler_enc.n_features_in_ = len(config['scaler_enc_min'])
        if 'scaler_enc_min_' in config:
            self.scaler_enc.min_ = config['scaler_enc_min_']
        else:
            self.scaler_enc.min_ = -config['scaler_enc_min'] * config['scaler_enc_scale']

        # Reconstruct decoder scaler
        self.scaler_dec = MinMaxScaler(feature_range=(0, 1))
        self.scaler_dec.data_min_ = config['scaler_dec_min']
        self.scaler_dec.data_max_ = config['scaler_dec_max']
        self.scaler_dec.scale_ = config['scaler_dec_scale']
        self.scaler_dec.data_range_ = config['scaler_dec_data_range']
        self.scaler_dec.n_features_in_ = len(config['scaler_dec_min'])
        if 'scaler_dec_min_' in config:
            self.scaler_dec.min_ = config['scaler_dec_min_']
        else:
            self.scaler_dec.min_ = -config['scaler_dec_min'] * config['scaler_dec_scale']

        # Reconstruct target scaler
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y.data_min_ = config['scaler_y_min']
        self.scaler_y.data_max_ = config['scaler_y_max']
        self.scaler_y.scale_ = config['scaler_y_scale']
        self.scaler_y.data_range_ = config['scaler_y_data_range']
        self.scaler_y.n_features_in_ = len(config['scaler_y_min'])
        if 'scaler_y_min_' in config:
            self.scaler_y.min_ = config['scaler_y_min_']
        else:
            self.scaler_y.min_ = -config['scaler_y_min'] * config['scaler_y_scale']

        # Training metrics for reference
        self.cv_r2 = float(config['cv_mean_r2'])
        self.cv_mae = float(config['cv_mean_mae'])
        self.cv_rmse = float(config['cv_mean_rmse'])

        print(f"Configuration loaded from: {self.config_path}")

    def prepare_encoder_input(self, df):
        """
        Prepare encoder input from historical data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data with required columns (at least lookback_steps rows)

        Returns:
        --------
        encoder_X : np.array (1, lookback_steps, n_encoder_features)
        """
        required = ['DateTime', 'Consumption', 'Temperature', 'Temperature_Predicted']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(df) < self.lookback_steps:
            raise ValueError(f"Need at least {self.lookback_steps} rows, got {len(df)}")

        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Hour'] = df['DateTime'].dt.hour
        df['Minute'] = df['DateTime'].dt.minute
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek

        # Cyclical encodings
        if 'Hour_sin' not in df.columns:
            df['Hour_sin'] = np.sin(2 * np.pi * (df['Hour'] + df['Minute']/60) / 24)
        if 'Hour_cos' not in df.columns:
            df['Hour_cos'] = np.cos(2 * np.pi * (df['Hour'] + df['Minute']/60) / 24)
        if 'DayOfWeek_sin' not in df.columns:
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        if 'DayOfWeek_cos' not in df.columns:
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        if 'IsWeekend' not in df.columns:
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        # Get last lookback_steps rows
        input_data = df.tail(self.lookback_steps)
        encoder_X = input_data[self.encoder_features].values

        # Normalize
        encoder_X_scaled = self.scaler_enc.transform(encoder_X)
        encoder_X_scaled = encoder_X_scaled.reshape(1, self.lookback_steps, self.n_encoder_features)

        return encoder_X_scaled, df

    def prepare_decoder_input(self, df, temp_forecast=None):
        """
        Prepare decoder input for future timesteps.

        The decoder needs:
        - Future known features (time encodings, predicted temperature)
        - Shifted target (last known consumption, then predictions during inference)

        Parameters:
        -----------
        df : pd.DataFrame
            Historical data (for getting last consumption value)
        temp_forecast : np.array (optional)
            Temperature forecast for next forecast_horizon steps
            If None, uses last known Temperature_Predicted

        Returns:
        --------
        decoder_X : np.array (1, forecast_horizon, n_decoder_features)
        timestamps : list of future timestamps
        """
        # Get last timestamp and consumption
        last_timestamp = pd.to_datetime(df['DateTime'].iloc[-1])
        last_consumption = df['Consumption'].iloc[-1]

        # Generate future timestamps
        future_timestamps = [last_timestamp + timedelta(minutes=5 * (i + 1))
                           for i in range(self.forecast_horizon)]

        # Create future time features
        future_df = pd.DataFrame({'DateTime': future_timestamps})
        future_df['Hour'] = future_df['DateTime'].dt.hour
        future_df['Minute'] = future_df['DateTime'].dt.minute
        future_df['DayOfWeek'] = future_df['DateTime'].dt.dayofweek

        future_df['Hour_sin'] = np.sin(2 * np.pi * (future_df['Hour'] + future_df['Minute']/60) / 24)
        future_df['Hour_cos'] = np.cos(2 * np.pi * (future_df['Hour'] + future_df['Minute']/60) / 24)
        future_df['DayOfWeek_sin'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
        future_df['DayOfWeek_cos'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
        future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)

        # Temperature forecast
        if temp_forecast is not None:
            if len(temp_forecast) != self.forecast_horizon:
                raise ValueError(f"temp_forecast must have {self.forecast_horizon} values")
            future_df['Temperature_Predicted'] = temp_forecast
        else:
            # Use last known predicted temperature (simple fallback)
            future_df['Temperature_Predicted'] = df['Temperature_Predicted'].iloc[-1]

        # Extract decoder features (without shifted target)
        decoder_features_no_target = [f for f in self.decoder_features if f != 'shifted_target']
        future_features = future_df[decoder_features_no_target].values

        # Create shifted target placeholder
        # During inference, we start with last known consumption
        # The model will autoregressively fill in predictions
        shifted_target = np.zeros(self.forecast_horizon)
        shifted_target[0] = last_consumption

        # For inference, we use the last known value for all positions
        # The model was trained with teacher forcing, but during inference
        # we use a simple approach: set all shifted targets to last known value
        # A more sophisticated approach would be autoregressive decoding
        shifted_target[:] = last_consumption

        # Combine decoder features with shifted target
        decoder_X = np.column_stack([future_features, shifted_target])

        # Normalize
        decoder_X_scaled = self.scaler_dec.transform(decoder_X)
        decoder_X_scaled = decoder_X_scaled.reshape(1, self.forecast_horizon, self.n_decoder_features)

        return decoder_X_scaled, future_timestamps

    def forecast(self, df, temp_forecast=None, return_timestamps=True):
        """
        Generate 2-day ahead consumption forecast.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data with at least lookback_steps rows
        temp_forecast : np.array (optional)
            Temperature forecast for future period
        return_timestamps : bool
            Whether to return forecast timestamps

        Returns:
        --------
        predictions : np.array
            Predicted consumption values (forecast_horizon values)
        timestamps : list (optional)
            Timestamps for each prediction
        """
        # Prepare encoder input (historical data)
        encoder_X, df_processed = self.prepare_encoder_input(df)

        # Prepare decoder input (future features)
        decoder_X, timestamps = self.prepare_decoder_input(df_processed, temp_forecast)

        # Predict
        y_pred_scaled = self.model.predict([encoder_X, decoder_X], verbose=0)

        # Inverse transform
        predictions = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        if return_timestamps:
            return predictions, timestamps
        return predictions

    def forecast_autoregressive(self, df, temp_forecast=None, return_timestamps=True):
        """
        Generate forecast using autoregressive decoding.

        This is more accurate but slower - it feeds each prediction back
        as input for the next step.

        Note: This requires modifying the model architecture to support
        step-by-step decoding. For now, this is a placeholder that uses
        the standard forecast method.
        """
        # For full autoregressive decoding, we would need to:
        # 1. Build separate encoder and decoder models
        # 2. Run encoder once to get context
        # 3. Run decoder step-by-step, feeding predictions back

        # For now, use standard inference
        return self.forecast(df, temp_forecast, return_timestamps)

    def plot_forecast(self, predictions, timestamps, actual=None, title=None):
        """Plot the forecast results."""
        fig, ax = plt.subplots(figsize=(16, 6))

        x_vals = np.arange(len(predictions))

        if actual is not None:
            ax.plot(x_vals, actual, 'b-', label='Actual', linewidth=1.5)
            ax.plot(x_vals, predictions, 'r--', label='Predicted', linewidth=1.5)
            ax.fill_between(x_vals, actual, predictions, alpha=0.3, color='gray')
        else:
            ax.plot(x_vals, predictions, 'b-', label='Forecast', linewidth=1.5)
            ax.fill_between(x_vals, predictions, alpha=0.3, color='steelblue')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Consumption (kW)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'2-Day Consumption Forecast (Seq2Seq Attention)\n{timestamps[0]} to {timestamps[-1]}',
                        fontsize=14, fontweight='bold')

        # Add tick labels every 2 hours
        tick_positions = np.arange(0, len(timestamps), 24)
        tick_labels = [timestamps[i].strftime('%m-%d %H:%M') for i in tick_positions if i < len(timestamps)]
        ax.set_xticks(tick_positions[:len(tick_labels)])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return fig

    def get_model_info(self):
        """Return model information and performance metrics."""
        return {
            'model_type': 'Seq2Seq Encoder-Decoder with Attention',
            'lookback_steps': self.lookback_steps,
            'forecast_horizon': self.forecast_horizon,
            'encoder_features': self.encoder_features,
            'decoder_features': self.decoder_features,
            'cv_r2': self.cv_r2,
            'cv_mae': self.cv_mae,
            'cv_rmse': self.cv_rmse
        }


def load_recent_data(filepath, n_rows=None):
    """
    Load recent data from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to data file
    n_rows : int (optional)
        Number of most recent rows to load

    Returns:
    --------
    df : pd.DataFrame
        Loaded and preprocessed data
    """
    data = pd.read_csv(filepath, skiprows=3, header=None, encoding='latin-1')

    data.columns = ['DateTime_str', 'Date', 'DayTime', 'Forecast_Prod', 'Forecast_Load',
                    'Consumption', 'Production', 'Level_Bidmi', 'Level_Haselholz',
                    'Temperature', 'Irradiance', 'Rain', 'SDR_Mode', 'Forecast_Mode',
                    'Transfer_Mode', 'Waterlevel_Mode', 'Temp_Forecast']

    data['DateTime'] = pd.to_datetime(data['DateTime_str'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    data = data.dropna(subset=['DateTime'])
    data = data.sort_values('DateTime').reset_index(drop=True)

    data['Temp_Forecast'] = pd.to_numeric(data['Temp_Forecast'], errors='coerce')
    data['Temp_Forecast'] = data['Temp_Forecast'].fillna(data['Temperature'])
    data.rename(columns={'Temp_Forecast': 'Temperature_Predicted'}, inplace=True)

    if n_rows:
        data = data.tail(n_rows)

    return data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SEQ2SEQ ATTENTION FORECASTER APPLICATION")
    print("=" * 80)

    model_path = 'Cons_Seq2Seq_Attention_Model.keras'
    config_path = 'Cons_Seq2Seq_Attention_Model_Config.npz'

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("\nModel not found. Please run Cons_Seq2Seq_Attention_Forecast.py first to train the model.")
        print(f"Expected files:")
        print(f"  - {model_path}")
        print(f"  - {config_path}")
    else:
        # Initialize forecaster
        print("\n1. Initializing forecaster...")
        forecaster = Seq2SeqForecaster(model_path, config_path)

        # Load recent data
        print("\n2. Loading recent data...")
        data = load_recent_data('../Data_January.csv', n_rows=1000)
        print(f"   Loaded {len(data)} rows")
        print(f"   Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

        # Generate forecast
        print("\n3. Generating 2-day forecast...")
        predictions, timestamps = forecaster.forecast(data)

        print(f"\n   Forecast Summary:")
        print(f"   Period: {timestamps[0]} to {timestamps[-1]}")
        print(f"   Mean consumption: {predictions.mean():.2f} kW")
        print(f"   Min consumption: {predictions.min():.2f} kW")
        print(f"   Max consumption: {predictions.max():.2f} kW")
        print(f"   Total energy (2 days): {predictions.sum() * 5 / 60:.2f} kWh")

        # Check for variability
        print(f"\n   Forecast Variability:")
        print(f"   Std deviation: {predictions.std():.2f} kW")
        print(f"   Range: {predictions.max() - predictions.min():.2f} kW")

        # Save forecast
        forecast_df = pd.DataFrame({
            'DateTime': timestamps,
            'Predicted_Consumption_kW': predictions
        })
        forecast_df.to_csv('Seq2Seq_Application_Forecast.csv', index=False)
        print(f"\n   Forecast saved to: Seq2Seq_Application_Forecast.csv")

        # Plot forecast
        print("\n4. Plotting forecast...")
        forecaster.plot_forecast(predictions, timestamps,
                                title='2-Day Ahead Consumption Forecast (Seq2Seq Attention)')

        # Print model info
        print("\n5. Model Information:")
        info = forecaster.get_model_info()
        print(f"   Model type: {info['model_type']}")
        print(f"   Lookback: {info['lookback_steps']} steps")
        print(f"   Forecast horizon: {info['forecast_horizon']} steps")
        print(f"   CV R2: {info['cv_r2']:.4f}")
        print(f"   CV MAE: {info['cv_mae']:.2f} kW")
        print(f"   CV RMSE: {info['cv_rmse']:.2f} kW")

        print("\n" + "=" * 80)
        print("FORECASTING COMPLETE")
        print("=" * 80)