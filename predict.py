import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model_path = 'trained_model.pkl'
regressor = joblib.load(model_path)

# Load new data
file_path = input("Enter the path to the new data file: ").strip()  # Remove leading/trailing spaces
new_data = pd.read_csv(file_path)

# Define numerical and categorical features
numerical_features = [ "cadastral_income","surface_land_sqm", "total_area_sqm","construction_year", "latitude", "longitude", "garden_sqm", "primary_energy_consumption_sqm",  "nbr_frontages", "nbr_bedrooms", "terrace_sqm"]
fl_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace", "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]

cat_features = ['property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 'state_building', 'epc', 'heating_type', 'equipped_kitchen']

# Impute missing values for numerical features
imputer = SimpleImputer(strategy="mean")
new_data[numerical_features] = imputer.fit_transform(new_data[numerical_features])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(new_data[cat_features]).toarray()

# Concatenate numerical, boolean, and encoded categorical features
X = np.concatenate([new_data[numerical_features], new_data[fl_features], encoded_features], axis=1)

# Print the shape of X
print("Shape of X:", X.shape)

# Make predictions
predictions = regressor.predict(X)

# Save predictions to a CSV file
output_file = 'predictions.csv'
pd.DataFrame(predictions, columns=['Predicted_Price']).to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
