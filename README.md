# Aircraft-Engine-Remaining-Useful-Life-RUL-Prediction
## Overview
This project develops and evaluates machine learning models to predict the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS dataset (FD001–FD004). The models aim to help in predictive maintenance by identifying how many operational cycles remain before engine failure.

Four datasets are used:

FD001 – Single operating condition, single fault mode

FD002 – Multiple operating conditions, single fault mode

FD003 – Single operating condition, multiple fault modes

FD004 – Multiple operating conditions, multiple fault modes

## Project Objectives
Predict RUL with high accuracy across different engine operating and fault conditions.

Compare the performance of multiple ML models:

Ridge Regression (Linear Regression with regularization)

Random Forest

Support Vector Regression (SVR)

CNN + LSTM hybrid model

## Dataset Description
Source: NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

Features:

21 sensor measurements recorded at each time step per engine unit

After feature selection, 7 low-information sensors (1, 5, 6, 10, 16, 18, 19) removed

14 sensors retained for model input

Labels: Remaining Useful Life (in cycles) for each engine instance

## Data Preprocessing
Feature Selection – Removal of low-information sensors.

Normalization – Min-Max scaling to the range [-1, 1].

Time Window Processing –

Window size: 40 time steps

For each time step, concatenate readings from the previous 40 steps → 560 features per sample.

Train/Test Split – Per dataset, engines are split into training and test sets.

## Modeling Approach
1. Ridge Regression
Ridge regularization with cross-validation for alpha tuning.

Best α found per dataset.

| Dataset | Mean RMSE | Test RMSE |
| ------- | --------- | --------- |
| FD001   | 33.6405   | 29.7764   |
| FD002   | 38.3264   | 31.6818   |
| FD003   | 53.9055   | 40.8188   |
| FD004   | 56.5306   | 42.2027   |


2. Random Forest
Tuned over max_depth and n_estimators.

Best parameters often: max_depth=30, n_estimators=50.

| Dataset | Mean RMSE | Test RMSE |
| ------- | --------- | --------- |
| FD001   | 32.6202   | 23.0173   |
| FD002   | 36.2204   | 28.4774   |
| FD003   | 44.6834   | 33.0915   |
| FD004   | 50.4060   | 35.2479   |


3. Support Vector Regression (SVR)
Kernel: RBF

Tuned C and epsilon.

Best config example: C=50, epsilon=1.
| Dataset | Validation RMSE | Test RMSE |
| ------- | --------------- | --------- |
| FD001   | 28.0879         | 21.7324   |
| FD002   | 42.7014         | 36.1147   |
| FD003   | 45.0460         | 27.1409   |
| FD004   | 62.0252         | 41.0824   |

4. CNN + LSTM Hybrid Model
Input: Shape (14 sensors, 40 timesteps, 1 channel)

CNN Layers:

Conv2D(16) → Conv2D(32) (or Conv2D(32) → Conv2D(64) in variant)

Kernel size: 3×1, Pool size: 2×1

LSTM Layer: Hidden size 32

Fully connected output layer for RUL prediction.

CNN (16, 32) + LSTM (32)
Epochs: 30

Batch size: 64

FD001 Test RMSE: 19.4297

FD002–FD004: To be filled

CNN (32, 64) + LSTM (32)
Epochs: 30

Batch size: 64

FD001 Test RMSE: 18.9118

FD002–FD004: To be filled


## Key Insights
SVR generally achieved the lowest RMSE for FD001 and FD003.

Random Forest performed well on FD002 and FD004.

CNN+LSTM showed promising results for FD001 and is expected to improve further with hyperparameter tuning for other datasets.

Datasets with multiple operating conditions (FD002, FD004) and multiple fault modes (FD003, FD004) are more challenging, leading to higher RMSE.

## References
NASA C-MAPSS Dataset: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

Livieris, I.E., Pintelas, E., & Pintelas, P. (2020). A CNN–LSTM model for time-series forecasting. Neural Computing & Applications, 32, 17351–17360.

Zhang, C., Lim, P., Qin, A.K., & Tan, K.C. (2016). Multiobjective deep belief networks ensemble for RUL estimation. IEEE Transactions on Neural Networks and Learning Systems.

