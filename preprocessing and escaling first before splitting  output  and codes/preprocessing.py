

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# %%
# Construct the relative path to your CSV file
csv_file_path = 'C:\\Users\\afshi\\Documents\\GitHub\\immo-eliza-ml\\data\\properties.csv'

# Read the CSV file and create a DataFrame
df = pd.read_csv(csv_file_path)
df.head()

    
    

# %%
print(df.dtypes)
df.columns



# %%

# Imputing missing values with mean in numerical values
numerical_features = ["cadastral_income","surface_land_sqm","price", "total_area_sqm", "latitude", "longitude", "garden_sqm", "primary_energy_consumption_sqm", "construction_year", "nbr_frontages", "nbr_bedrooms", "terrace_sqm" ]
fl_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace","fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
cat_features = ['property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 'state_building', 'epc', 'heating_type', 'equipped_kitchen']



imputer = SimpleImputer(strategy='mean')
df[numerical_features] = imputer.fit_transform(df[numerical_features])
nan_counts = df.isna().sum()
print(nan_counts)
df.head()






# %%

## Imputing missing values on categorical values
# Replace "MISSING" with None in categorical features
df[cat_features] = df[cat_features].replace("MISSING", None)

# Drop missing values from the entire DataFrame
df.dropna(inplace=True)

# Count missing values in the entire DataFrame
nan_counts = df.isna().sum()

# Print the count of missing values
print(nan_counts)


# %%
# Imputing missing values on f1 features values
df[fl_features].isnull().sum()


# %%
# Perform one-hot encoding for categorical features
#df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)

# Print the encoded DataFrame
#print(df_encoded.head())






# Assuming 'df' is your DataFrame and 'cat_features' contains categorical feature names

# Perform one-hot encoding for categorical features


# Assuming 'df' is your DataFrame and 'cat_features' contains categorical feature names

# Verify Categorical Features List
# Perform one-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)
# Convert boolean values to integers (0s and 1s)
df_encoded = df_encoded.astype(int)



# Print the first few rows of the encoded DataFrame to verify
print(df_encoded.head())








# %%

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the numeric features
df_encoded[numerical_features] = scaler.fit_transform(df_encoded [numerical_features])

# Print the standardized DataFrame
print(df_encoded .head())





# %%
# Save df_encoded to a CSV file
df_encoded.to_csv('encoded_data.csv', index=False)



