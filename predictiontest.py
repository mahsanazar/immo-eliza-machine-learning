import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Preprocess the input data.
    
    Args:
    - df (pd.DataFrame): Input DataFrame.
    
    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    numerical_data = df[numerical_features].values
    imputer = SimpleImputer(strategy="mean")
    numerical_data = imputer.fit_transform(numerical_data)
    
    categorical_data = df[categorical_features]
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_data_encoded = encoder.fit_transform(categorical_data)
    
    processed_data = np.concatenate([numerical_data, categorical_data_encoded.toarray()], axis=1)
    
    return processed_data

def load_model(model_path):
    """
    Load the trained model from the specified path.
    
    Args:
    - model_path (str): Path to the trained model file.
    
    Returns:
    - regressor: Trained model object.
    """
    regressor = joblib.load(model_path)
    return regressor

def make_predictions(model, data):
    """
    Make predictions using the trained model.
    
    Args:
    - model: Trained model object.
    - data (pd.DataFrame): DataFrame containing input data.
    
    Returns:
    - np.array: Array of predicted prices.
    """
    predictions = model.predict(data)
    return predictions

def main():
    # Load the trained model
    model_path = 'trained_model.pkl'
    regressor = load_model(model_path)
    
    # Load new data
    file_path = input("Enter the path to the new data file: ")
    new_data = pd.read_csv(file_path)
    
    # Preprocess the new data
    processed_data = preprocess_data(new_data)
    
    # Make predictions
    predictions = make_predictions(regressor, processed_data)
    
    # Display predictions
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
