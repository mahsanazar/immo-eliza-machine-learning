# Real Estate Price Prediction

## Overview

This project aims to predict real estate prices based on various features such as property type, location, and amenities. It includes two main scripts:


1. **train.py**: This script is used to train a machine learning model on the provided real estate dataset.
2. **predict.py**: After training the model, you can use this script to make predictions on new real estate data.
Repository: immo-eliza-ml
Duration: 5 days
Team: solo

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- joblib

## Installation

1. Clone the repository


2. Navigate to the project directory


3. Install the required dependencies


## Usage

### Training the Model (train.py)

Run the train.py script to train the machine learning model on the real estate dataset

python train.py

This script performs the following steps:

- Load the real estate dataset from the data/properties.csv file.
- Preprocess the data, including handling missing values and encoding categorical features.
- Split the dataset into training and testing sets.
- Train a linear regression model on the training data.
- Evaluate the trained model on the testing data.
- Visualize the actual vs. predicted prices for both training and testing sets.
- Save the trained model to a file named trained_model.pkl.

### Making Predictions (predict.py)

Once the model is trained and saved, you can use the predict.py script to make predictions on new real estate data:

python predict.py

This script prompts you to enter the path to the new data file containing the real estate properties for which you want to make predictions. After entering the file path, it:

- Loads the trained model from the trained_model.pkl file.
- Loads the new data from the specified CSV file.
- Preprocesses the new data using the same preprocessing steps as the training data.
- Makes predictions using the trained model.
- Saves the predictions to a CSV file named predictions.csv  and excel file named predictions.xlsx.


## Folder Structure

- **data**: Contains the real estate dataset (properties.csv) and any additional data files.
- **visualizations**: Contains visualizations of the actual vs. predicted prices.
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **MODELSCARD.md**: Provides information about the trained model, including its performance metrics and features.
- **README.md**: Main documentation file for the project.
- **notebook.ipynb**: Jupyter notebook for exploratory data analysis or additional modeling.
- **predict.py**: Script for making predictions on new real estate data.
- **predictiontest.py**: Script for testing predictions with different input with different columns and features.
- **train with linerregression.py**: Script for training the machine learning model by linerregression and preprocessing.
- **train with crossvalidation.py**: Script for training the machine learning model by linerregression and calculation of crossvalidation
- **trained_model.pkl**: Serialized trained model saved for future use.
- **randomforest**:Script for training the machine learning model by randomforest
- **catboost**:Script for training the machine learning model by catboost
- **scores**:all of scores : test and train and correlation scores are stored in this file
- **preprocessing**:Includes scripts or notebooks for experimenting with data preprocessing and feature scaling using various methods




## Author

- Mahsa Nazarian
- GitHub: (https:https://github.com/mahsanazar)


You can copy and paste this content directly into your README.md file. 

## Comments
I  tested other models such as randomforest and catboost; their codes  , their visualizations, their MODELSCARD and  scores and predictions are stored in seperate folders, the scores of catboost model was really better than linerregression. The randomforest had a big file and could not be pushed.

I've explored different approaches in the preprocessing folder, but the main training and preprocessing tasks are consolidated in train.py. Please disregard the preprocessing folder and navigate to the train.py folder instead.
