import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import openpyxl

# Load the dataset
github_url = "https://raw.githubusercontent.com/your-username/your-repo/main/AmesHousing.xlsx"
df = pd.read_excel(github_url, engine='openpyxl')

# Data preprocessing (handle missing values, select features, encode categorical data)
df = df.select_dtypes(include=[np.number]).dropna()
if 'SalePrice' not in df.columns:
    st.error("Error: 'SalePrice' column is missing from the dataset.")
    st.stop()

X = df.drop(columns=['SalePrice'])  # Features
y = df['SalePrice']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit Web App
st.title("Housing Price Prediction")
st.write("Enter house features to predict the sale price.")

input_features = {}
for feature in X.columns:
    input_features[feature] = st.number_input(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_features])
    input_scaled = scaler.transform(input_df)
    predicted_price = model.predict(input_scaled)
    st.write(f"Predicted Sale Price: ${predicted_price[0]:,.2f}")

# Save necessary files for deployment
requirements = """pandas
numpy
streamlit
scikit-learn
openpyxl"""
with open("requirements.txt", "w") as f:
    f.write(requirements)

# Instructions:
# 1. Upload 'AmesHousing.xlsx' and this script to your GitHub repository.
# 2. Deploy the app using Streamlit by running 'streamlit run script_name.py'.
# 3. Update the 'github_url' variable with your actual GitHub raw file link.
