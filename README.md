# Thesis_AlpenEnergie

Cons_Prediction Folder is the Load Forecast Code predicting the 48 next hours electricity consumption in the municipality

Command for the CNN_LSTM Programm:
## Train the 48h model (default)
python CNN_LSTM_Prediction.py --train

## Train the 96h model
python CNN_LSTM_Prediction.py --train --hours 96

## Predict 48h from a date (uses CNN_LSTM_Model_48h.keras)
python CNN_LSTM_Prediction.py --predict 2026-03-13

## Predict 96h from a date (uses CNN_LSTM_Model_96h.keras)
python CNN_LSTM_Prediction.py --predict 2026-03-13 --hours 96

Worklow:

1. First Fetch the fresh Weather Data:
python get_weather_data.py

2. Train the Model if needed:
python CNN_LSTM_Prediction.py --train
python CNN_LSTM_Prediction.py --train --hours 96

3. Test a Prediction Manually:
python CNN_LSTM_Prediction.py --predict 2026-03-13
python CNN_LSTM_Prediction.py --predict 2026-03-14 --hours 96

4. Results available in the Result folder

Optimisation Folder is my Optimisation Code to Optimise the Production and Maximise the earnings. Battery should be implemented to asses its viability
