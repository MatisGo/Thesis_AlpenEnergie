""" 
CNN-LSTM Model Application Script
=================================
This script loads a trained CNN-LSTM model and performs 2-day ahead
consumption forecasting using:
1. Historical consumption data
2. Temperature forecasts

USE CASES:
----------
1. Real-time forecasting: Predict next 2 days based on latest data
2. What-if analysis: Test different temperature scenarios
3. Batch forecasting: Generate predictions for multiple starting points

REQUIREMENTS:
-------------
- Trained model: Cons_CNN_LSTM_Model.keras
- Configuration: Cons_CNN_LSTM_Model_Config.npz
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


class CNNLSTMForecaster:
    """
    Class for loading and using a trained CNN-LSTM model for consumption forecasting.
    """

    def __init__(self, model_path='Cons_CNN_LSTM_Model.keras', config_path='Cons_CNN_LSTM_Model_Config.npz'):
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

        # Load model and config
        self._load_model()
        self._load_config()

        print(f"CNN-LSTM Forecaster initialized")
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
        self.feature_columns = list(config['feature_columns'])
        self.n_features = int(config['n_features'])

        # Reconstruct scaler_X
        # MinMaxScaler needs: min_, scale_, data_min_, data_max_, data_range_, n_features_in_
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_X.data_min_ = config['scaler_X_min']
        self.scaler_X.data_max_ = config['scaler_X_max']
        self.scaler_X.scale_ = config['scaler_X_scale']
        self.scaler_X.data_range_ = config['scaler_X_data_range']
        self.scaler_X.n_features_in_ = len(config['scaler_X_min'])
        # min_ is stored directly if available, otherwise calculate it
        if 'scaler_X_min_' in config:
            self.scaler_X.min_ = config['scaler_X_min_']
        else:
            self.scaler_X.min_ = -config['scaler_X_min'] * config['scaler_X_scale']

        # Reconstruct scaler_y
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

    def prepare_data(self, df):
        """
        Prepare input data for forecasting.

        The input DataFrame must contain the required columns and have
        at least lookback_steps rows.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data with required columns

        Returns:
        --------
        X : np.array
            Prepared input features (1, lookback_steps, n_features)
        """
        # Check required columns
        required_base = ['DateTime', 'Consumption', 'Temperature', 'Temperature_Predicted']
        for col in required_base:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check data length
        if len(df) < self.lookback_steps:
            raise ValueError(f"Need at least {self.lookback_steps} rows, got {len(df)}")

        # Add time features if not present
        df = df.copy()
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df['Hour'] = df['DateTime'].dt.hour
            df['Minute'] = df['DateTime'].dt.minute
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek

        # Add cyclical encodings
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

        # Get the last lookback_steps rows
        input_data = df.tail(self.lookback_steps)

        # Extract features
        X = input_data[self.feature_columns].values

        # Normalize
        X_scaled = self.scaler_X.transform(X.reshape(-1, self.n_features))
        X_scaled = X_scaled.reshape(1, self.lookback_steps, self.n_features)

        return X_scaled

    def forecast(self, df, return_timestamps=True):
        """
        Generate 2-day ahead consumption forecast.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data with at least lookback_steps rows
        return_timestamps : bool
            Whether to return forecast timestamps

        Returns:
        --------
        predictions : np.array
            Predicted consumption values (forecast_horizon values)
        timestamps : list (optional)
            Timestamps for each prediction
        """
        # Prepare input
        X = self.prepare_data(df)

        # Predict
        y_pred_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler_y.inverse_transform(y_pred_scaled).flatten()

        if return_timestamps:
            # Generate future timestamps
            last_timestamp = pd.to_datetime(df['DateTime'].iloc[-1])
            timestamps = [last_timestamp + timedelta(minutes=5 * (i + 1))
                         for i in range(self.forecast_horizon)]
            return predictions, timestamps

        return predictions

    def forecast_with_temp_scenario(self, df, temp_forecast):
        """
        Generate forecast with a specific temperature scenario.

        This allows testing different temperature conditions.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data (will use Temperature_Predicted from this)
        temp_forecast : np.array
            Temperature forecast values to use (should match forecast_horizon)

        Returns:
        --------
        predictions : np.array
            Predicted consumption values
        timestamps : list
            Timestamps for predictions
        """
        # For now, the model uses historical data including Temperature_Predicted
        # To fully support temp scenarios, we would need to modify the architecture
        # to accept future temperature as a separate input

        # This is a simplified version - full implementation would require
        # architecture changes
        return self.forecast(df)

    def plot_forecast(self, predictions, timestamps, actual=None, title=None):
        """
        Plot the forecast results.

        Parameters:
        -----------
        predictions : np.array
            Predicted values
        timestamps : list
            Timestamps for predictions
        actual : np.array (optional)
            Actual values for comparison
        title : str (optional)
            Plot title
        """
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
            ax.set_title(f'2-Day Consumption Forecast\n{timestamps[0]} to {timestamps[-1]}',
                        fontsize=14, fontweight='bold')

        # Add tick labels every 2 hours
        tick_positions = np.arange(0, len(timestamps), 24)  # Every 2 hours (24 * 5min)
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
            'lookback_steps': self.lookback_steps,
            'forecast_horizon': self.forecast_horizon,
            'feature_columns': self.feature_columns,
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
    print("CNN-LSTM FORECASTER APPLICATION")
    print("=" * 80)

    # Check if model exists
    model_path = 'Cons_CNN_LSTM_Model.keras'
    config_path = 'Cons_CNN_LSTM_Model_Config.npz'

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("\nModel not found. Please run Cons_CNN_LSTM_Forecast.py first to train the model.")
        print(f"Expected files:")
        print(f"  - {model_path}")
        print(f"  - {config_path}")
    else:
        # Initialize forecaster
        print("\n1. Initializing forecaster...")
        forecaster = CNNLSTMForecaster(model_path, config_path)

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

        # Save forecast
        forecast_df = pd.DataFrame({
            'DateTime': timestamps,
            'Predicted_Consumption_kW': predictions
        })
        forecast_df.to_csv('Application_Forecast.csv', index=False)
        print(f"\n   Forecast saved to: Application_Forecast.csv")

        # Plot forecast
        print("\n4. Plotting forecast...")
        forecaster.plot_forecast(predictions, timestamps,
                                title='2-Day Ahead Consumption Forecast')

        # Print model info
        print("\n5. Model Information:")
        info = forecaster.get_model_info()
        print(f"   Lookback: {info['lookback_steps']} steps")
        print(f"   Forecast horizon: {info['forecast_horizon']} steps")
        print(f"   CV R²: {info['cv_r2']:.4f}")
        print(f"   CV MAE: {info['cv_mae']:.2f} kW")
        print(f"   CV RMSE: {info['cv_rmse']:.2f} kW")

        print("\n" + "=" * 80)
        print("FORECASTING COMPLETE")
        print("=" * 80)
