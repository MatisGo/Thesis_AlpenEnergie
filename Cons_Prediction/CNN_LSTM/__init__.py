"""
Deep Learning Models for Consumption Forecasting
=================================================

This package contains deep learning models for 2-day ahead
electricity consumption forecasting.

AVAILABLE MODELS:
-----------------

1. CNN-LSTM (Direct Multi-Step)
   - Cons_CNN_LSTM_Forecast.py: Training script
   - CNN_LSTM_Application.py: Inference script
   - Predicts all 576 steps (48h) at once
   - Fast inference, but may produce flat predictions

2. Seq2Seq Encoder-Decoder with Attention (RECOMMENDED)
   - Cons_Seq2Seq_Attention_Forecast.py: Training script
   - Seq2Seq_Attention_Application.py: Inference script
   - Step-by-step prediction with attention mechanism
   - Better captures daily patterns and variability

OTHER MODULES:
--------------
- Data_Quality_Review.py: Data analysis and outlier detection

USAGE:
------
1. First run Data_Quality_Review.py to analyze data quality
2. Train a model:
   - For Seq2Seq (recommended): python Cons_Seq2Seq_Attention_Forecast.py
   - For CNN-LSTM: python Cons_CNN_LSTM_Forecast.py
3. Use application scripts for production forecasting

ARCHITECTURE REFERENCES:
------------------------
CNN-LSTM:
- Chung & Jang (2022): "Accurate prediction of electricity consumption
  using a hybrid CNN-LSTM model based on multivariable data"
- Khan et al. (2020): "Towards Efficient Electricity Forecasting in
  Residential and Commercial Buildings"

Seq2Seq with Attention:
- "Attention-Enhanced LSTM for Long-Horizon Time Series Forecasting" (2024)
- "Optimized Seq2Seq model for short-term power load forecasting" (2023)
"""

from .CNN_LSTM_Application import CNNLSTMForecaster, load_recent_data
from .Seq2Seq_Attention_Application import Seq2SeqForecaster

__version__ = '2.0.0'
__author__ = 'Master Thesis - AlpenEnergie'
