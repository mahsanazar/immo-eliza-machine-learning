# Model Card for CatBoostRegressor model

## Overview

This document provides information about the trained machine learning model for real estate price prediction.

## Model Details

- **Model Name**: CatBoostRegressor
- **Training Data**: Real estate dataset containing various features such as property type, location, and amenities.
- **Target Variable**: Price (real estate prices)
- **Features Used**: Numerical features (e.g., total area, number of bedrooms) and categorical features (e.g., property type, heating type).
- **Preprocessing**: Imputation of missing values using mean for numerical features, one-hot encoding for categorical features.
- **Evaluation Metric**: R-squared
- **Performance Metrics**:
  - Train Score: 0.90
  - Test Score: 0.77

## Model Usage

The trained model is used to predict real estate prices based on input data containing the same features as used during training. The `predict.py` script can be used to make predictions on new real estate data.

## Folder Structure


- **MODELSCARD.md**: This document providing information about the model.
- **train.py**: Python script for training the machine learning model.
- **trained_catboost_model.pkl**: Pickled file containing the trained CatBoost model.
- **figure_1 and figure_2**: The visualization of  predictions and real data on training and test dataset

## Author

Mahsa Nazarian



