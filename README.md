# Thesis_AlpenEnergie

Python Version 3.12

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


Command the Prod_Prediction programm:

## Train the 48h model (default)
python DNN_Prod_Prediction.py --train

## Train the 96h  model 
python DNN_Prod_Prediction.py --train-weekend

## Train the intraday shape Model
python DNN_Prod_Prediction.py --train-shape

## Predict a certain day or a full Weekend (auto-selects model by weekday)
python DNN_Prod_Prediction.py --predict      

The Following Worklow is now automatise in the main.py script which can be started easily with the Run_forecast.bat. Please Read the SETUP.txt for the instructions


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
