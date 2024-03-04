import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler  # Add this import line
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the preprocessed data
df = pd.read_csv('C:/Users/demonking/Downloads/demand forecasting dataset/SalesFINAL12312016.csv')

# Convert 'SalesDate' to datetime for better handling
df['SalesDate'] = pd.to_datetime(df['SalesDate'])

# Extract month and year from 'SalesDate'
df['Month'] = df['SalesDate'].dt.month
df['Year'] = df['SalesDate'].dt.year

# Group by InventoryId, Store, Month, and Year and aggregate the required values
grouped_df = df.groupby(['InventoryId', 'Store', 'Month', 'Year']).agg({
    'SalesQuantity': 'sum',
    'SalesDollars': 'sum',
    'Volume': 'sum'
}).reset_index()

# Merge the grouped_df with the original df to add new columns
df = pd.merge(df, grouped_df, on=['InventoryId', 'Store', 'Month', 'Year'], how='left', suffixes=('', '_grouped'))

# Create new columns in the original df
df['MonthlyQuantity'] = df['SalesQuantity_grouped']
df['MonthlyDollars'] = df['SalesDollars_grouped']
df['MonthlyVolume'] = df['Volume_grouped']

# Save the updated dataframe to a new CSV file
df.to_csv('C:/Users/demonking/Downloads/demand forecasting dataset/SalesUpdated.csv', index=False)

df = df.drop(['SalesQuantity_grouped', 'SalesDollars_grouped', 'Volume_grouped'], axis=1)

# Convert 'SalesDate' to datetime for better handling
df['SalesDate'] = pd.to_datetime(df['SalesDate'])

# Extract month and year from 'SalesDate'
df['Month'] = df['SalesDate'].dt.month
df['Year'] = df['SalesDate'].dt.year

df.drop('SalesDate', axis=1, inplace=True)

def handle_outliers(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

# Specify the numeric columns for which you want to handle outliers
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Call the handle_outliers function to handle outliers in numeric columns
handle_outliers(df, numeric_columns)

def predict_season_binary(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn

df['Season'] = df['Month'].apply(predict_season_binary)

X = df[['Store', 'Brand', 'Month', 'Year']]
y = df["MonthlyQuantity"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, max_depth=None, max_features=10, min_samples_split=2, bootstrap=True, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

import joblib

# Save the trained model to a file
joblib.dump(model, 'trained_model.joblib')

# # Function to predict monthly quantity
# def predict_monthly_quantity(store, brand, month, year):
#     input_data = pd.DataFrame({
#         'Store': [store],
#         'Brand': [brand],
#         'Month': [month],
#         'Year': [year]
#     })

#     prediction = model.predict(input_data)
#     return prediction[0]

# Streamlit app
# st.title('Monthly Quantity Prediction App')

# # User input
# store = st.selectbox('Select Store', df['Store'].unique())
# brand = st.selectbox('Select Brand', df['Brand'].unique())
# month = st.slider('Select Month', 1, 12, 1)
# selected_year = st.text_input('Enter Year', df['Year'].min())

# # Convert the entered year to an integer
# year = int(selected_year)
# # Predict button
# if st.button('Predict Monthly Quantity'):
#     # Make prediction
#     prediction = predict_monthly_quantity(store, brand, month, year)

#     # Display the result
#     st.success(f'Predicted Monthly Quantity: {prediction:.2f}')
