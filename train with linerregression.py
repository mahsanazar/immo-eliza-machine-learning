# Load libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import joblib


# Load the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\afshi\\Documents\\GitHub\\immo-eliza-ml\\data\\properties.csv')

# Define features to use


numerical_features = [ "cadastral_income","surface_land_sqm", "total_area_sqm","construction_year", "latitude", "longitude", "garden_sqm", "primary_energy_consumption_sqm",  "nbr_frontages", "nbr_bedrooms", "terrace_sqm"]
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
imputer.fit(X_train[numerical_features])
X_train[numerical_features] = imputer.transform(X_train[numerical_features])
X_test[numerical_features] = imputer.transform(X_test[numerical_features])


# Initialize OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')# it ignores if it cannot convert
#enc = OneHotEncoder()

# Fit OneHotEncoder to training data
enc.fit(X_train[cat_features])

# Transform categorical features in training and testing data
X_train_cat = enc.transform(X_train[cat_features]).toarray()
X_test_cat = enc.transform(X_test[cat_features]).toarray()


# Concatenate numerical features, encoded categorical features, and f1_features for training and testing data
X_train_processed = np.concatenate([X_train[numerical_features], X_train_cat, X_train[fl_features]], axis=1)
X_test_processed = np.concatenate([X_test[numerical_features], X_test_cat, X_test[fl_features]], axis=1)

# Plot each feature against the target variable

# Plot each feature against the target variable
# List of all feature names
all_features = numerical_features +  cat_features

# Plot each feature against the target variable
#for i, feature_name in enumerate(all_features):
#    plt.figure(figsize=(8, 6))
#    plt.scatter(X_train_processed[:, i], y_train, c='blue', alpha=0.5)
#    plt.xlabel(feature_name)
#    plt.ylabel('Price')
#    plt.title('{} vs Target'.format(feature_name))
#    plt.grid(True)
#    plt.show()




# Calculate correlation coefficients for numerical features
numerical_correlation = pd.DataFrame(X_train_processed[:, :len(numerical_features)], columns=numerical_features).corrwith(y_train)

# Calculate correlation coefficients for encoded categorical features
#encoded_cat_correlation = pd.DataFrame(X_train_processed[:, len(numerical_features):], columns=enc.get_feature_names_out(cat_features)).corrwith(y_train)

print("Correlation Coefficients for Numerical Features:")
print(numerical_correlation)

#print("\nCorrelation Coefficients for Categorical Features:")
#print(encoded_cat_correlation)

# Calculate correlation coefficients for encoded categorical features
encoded_cat_correlation = pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out(cat_features)).corrwith(y_train)
#Calculate correlation coefficients for F1 features (boolean features)
f1_features_correlation = pd.DataFrame(X_train[fl_features], columns=fl_features).corrwith(y_train)
print("\nCorrelation Coefficients for Encoded Categorical Features:")
print(encoded_cat_correlation)

print("\nCorrelation Coefficients for F1 Features:")
print(f1_features_correlation)


# Train the model on the training data
regressor = LinearRegression()
regressor.fit(X_train_processed, y_train)

# Make predictions
y_train_pred = regressor.predict(X_train_processed)
y_test_pred = regressor.predict(X_test_processed)

# Calculate the scores
train_score = regressor.score(X_train_processed, y_train)
test_score = regressor.score(X_test_processed, y_test)
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


# Save the trained model
joblib.dump(regressor, 'trained_model.pkl')