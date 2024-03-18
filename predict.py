import pandas as pd
import numpy as np
import joblib

def preprocess_data(df):
    """
    Preprocess the input data.
    
    Args:
    - df (pd.DataFrame): Input DataFrame.
    
    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    numerical_features = ["cadastral_income", "surface_land_sqm", "total_area_sqm", "construction_year", 
                          "latitude", "longitude", "garden_sqm", "primary_energy_consumption_sqm",  
                          "nbr_frontages", "nbr_bedrooms", "terrace_sqm"]
    fl_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace", "fl_swimming_pool", 
                   "fl_floodzone", "fl_double_glazing"]
    cat_features = ['property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 
                    'state_building', 'epc', 'heating_type', 'equipped_kitchen']
    
    # Impute missing values using SimpleImputer on numerical features
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    df[numerical_features] = imputer.fit_transform(df[numerical_features])
    
    # Initialize OneHotEncoder
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    
    # Transform categorical features
    cat_data = enc.fit_transform(df[cat_features]).toarray()
    df = pd.concat([df[numerical_features], pd.DataFrame(cat_data), df[fl_features]], axis=1)
    
    return df

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


def save_predictions(predictions, output_file):
    """
    Save the predictions to a CSV file.
    
    Args:
    - predictions (np.array): Array of predicted prices.
    - output_file (str): Path to the output CSV file.
    """
    pd.DataFrame(predictions, columns=['Predicted_Price']).to_csv(output_file, index=False)

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
    
       # Save predictions to a CSV file
    output_file = 'predictions.csv'
    save_predictions(predictions, output_file)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
