# Load libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score 

# Load the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\afshi\\Documents\\GitHub\\immo-eliza-ml\\data\\properties.csv')

# Define features to use
numerical_features = ["cadastral_income", "surface_land_sqm", "total_area_sqm", "construction_year", "latitude", "longitude", "garden_sqm", "primary_energy_consumption_sqm", "nbr_frontages", "nbr_bedrooms", "terrace_sqm"]
fl_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace", "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
cat_features = ['property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 'state_building', 'epc', 'heating_type', 'equipped_kitchen']

# Split the data into features and target
X = df[numerical_features + fl_features + cat_features]
y = df["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Impute missing values using SimpleImputer on numerical features
imputer = SimpleImputer(strategy="mean")
X_train[numerical_features] = imputer.fit_transform(X_train[numerical_features])
X_test[numerical_features] = imputer.transform(X_test[numerical_features])

# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
# Fit and transform categorical features in training data
X_train_cat_encoded = encoder.fit_transform(X_train[cat_features]).toarray()
# Transform categorical features in testing data
X_test_cat_encoded = encoder.transform(X_test[cat_features]).toarray()

# Concatenate numerical features, encoded categorical features, and fl_features for training and testing data
X_train_processed = np.concatenate([X_train[numerical_features], X_train_cat_encoded, X_train[fl_features]], axis=1)
X_test_processed = np.concatenate([X_test[numerical_features], X_test_cat_encoded, X_test[fl_features]], axis=1)

# Train the CatBoost model on the training data
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function='RMSE', random_seed=42)
catboost_model.fit(X_train_processed, y_train)

# Make predictions
y_train_pred = catboost_model.predict(X_train_processed)
y_test_pred = catboost_model.predict(X_test_processed)

# Calculate the scores
train_score = r2_score(y_train, y_train_pred)
test_score = r2_score(y_test, y_test_pred)
print(f"Train score is: {train_score}")
print(f"Test score is: {test_score}")

# Visualize the actual vs. predicted prices for training set
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Predictions')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Actuals')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Training Set)')
plt.legend()
plt.show()

# Visualize the actual vs. predicted prices for test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actuals')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Test Set)')
plt.legend()
plt.show()

# Save the trained CatBoost model
joblib.dump(catboost_model, 'trained_catboost_model.pkl')
