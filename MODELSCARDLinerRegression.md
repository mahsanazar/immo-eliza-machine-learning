# Model Card for liner regression model

## Overview

This document provides information about the trained machine learning model for real estate price prediction.

## Model Details

- **Model Name**: Linear Regression
- **Training Data**: Real estate dataset containing various features such as property type, location, and amenities.
- **Target Variable**: Price (real estate prices)
- **Features Used**: Numerical features (e.g., total area, number of bedrooms) and categorical features (e.g., property type, heating type).
- **Preprocessing**: Imputation of missing values using mean for numerical features, one-hot encoding for categorical features.
- **Evaluation Metric**: R-squared (coefficient of determination and crossvalidation)
- **Performance Metrics**:
  - Train Score:0.46
  - Test Score:0.446

## Model Usage

The trained model is used to predict real estate prices based on input data containing the same features as used during training. The `predict.py` script can be used to make predictions on new real estate data.

## Limitations and Considerations

- The model performance may vary depending on the quality and completeness of the input data.
- It assumes a linear relationship between the features and the target variable, which may not always hold true.
- The model does not account for external factors such as economic conditions or market trends, which can influence real estate prices.

## Responsible AI Considerations

- Data Bias: Care should be taken to ensure that the training data is representative of the population and does not contain biases that could result in unfair predictions.
- Transparency: The model's inputs, preprocessing steps, and evaluation metrics are documented to provide transparency and enable users to understand its behavior.
- Accountability: Users should be aware of the model's limitations and uncertainties and use its predictions responsibly.

## Folder Structure

- **data**: Directory containing the dataset files.
- **preprocessing**: Code for data preprocessing and scaling before splitting.
- **output**: Directory for storing output files and visualizations.
- **visualizations**:  output files for visualizing predictions and real data on training and test dataset.
- **.gitignore**: File specifying which files and directories to ignore in version control.
- **MODELSCARD.md**: This document providing information about the model.
- **README.md**: Main README file for the project.
- **notebook.ipynb**: Jupyter notebook containing exploratory data analysis and model training code.
- **predict.py**: Python script for making predictions using the trained model.
- **predictiontest.py**: Python script for testing predictions.
- **train.py**: Python script for training the machine learning model.
- **trained_model.pkl**: Pickled file containing the trained model.

## Author

- Mahsa Nazarian
- GitHub: https://github.com/mahsanazar
## Improvement 
After testing various filters on the independent features, I found that the train score deteriorated. Additionally, experimenting with different scaling techniques led to a decrease in the test score. As a result, I opted not to alter the feature scaling and included all features in the dataset



