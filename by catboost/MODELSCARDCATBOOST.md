# Model Card for CatBoostRegressor model

## Overview

This document provides information about the trained machine learning model for real estate price prediction.

## Model Details

- **Model Name**: CatBoostRegressor
- **Training Data**: Real estate dataset containing various features such as property type, location, and amenities.
- **Target Variable**: Price (real estate prices)
- **Features Used**: Numerical features (e.g., total area, number of bedrooms) and categorical features (e.g., property type, heating type)
    and   F1_features.
- **Preprocessing**: Imputation of missing values using mean for numerical features, one-hot encoding for categorical features.
- **Evaluation Metric**: R-squared
- **Performance Metrics**:
  - Train Score: 0.90
  - Test Score: 0.77

## Model Usage

The trained model is used to predict real estate prices based on input data containing the same features as used during training. The `predictbycatboost.py` script can be used to make predictions on new real estate data.




## Author

Mahsa Nazarian



