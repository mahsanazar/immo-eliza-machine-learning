# Model Card: CatBoostRegressor

## Overview

The CatBoostRegressor model is a supervised learning algorithm used for predicting real-valued target variables. It is based on gradient boosting on decision trees and is particularly effective for regression tasks.

## Model Details

- **Model Type**: CatBoostRegressor (Gradient Boosting)
- **Model Version**: 1.0
- **Training Dataset**: Properties dataset
- **Features**: Cadastral income, surface land area, total area, construction year, latitude, longitude, garden area, primary energy consumption, number of frontages, number of bedrooms, terrace area, etc.
- **Target Variable**: Price
- **Preprocessing**: Missing values imputed with mean, categorical features one-hot encoded, no feature scaling applied

## Performance Metrics

- **Training  Score**: 0.90
- **Testing  Score**: 0.77


## Usage Instructions

1. **Training**: Run the `train with catboost.py` script to train the CatBoostRegressor model on the provided properties dataset. Ensure that the required libraries are installed (`pandas`, `numpy`, `catboost`, `scikit-learn`).

2. Prediction: Run the predictbycatboost.py script to make predictions using the trained model on new data. Provide the path to the new data file when prompted. The predictions will be saved to both CSV and Excel files.



