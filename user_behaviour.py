import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and necessary objects
model = joblib.load("best_model.pkl")  # Assuming the model is saved as best_model.pkl
scaler = joblib.load("scaler.pkl")  # Assuming the scaler is saved as scaler.pkl
label_encoder = joblib.load("label_encoder.pkl")  # Assuming the label encoder is saved as label_encoder.pkl

# Streamlit input form for user to enter values
st.title('User Behavior Prediction')
st.write("Enter the details to predict the User Behavior Class:")

# Input fields for user inputs
age = st.number_input("Age", min_value=18, max_value=100, step=1)
app_usage_time = st.number_input("App Usage Time (min/day)", min_value=0, step=1)
screen_on_time = st.number_input("Screen On Time (hours/day)", min_value=0, step=1)
battery_drain = st.number_input("Battery Drain (mAh/day)", min_value=0, step=1)
num_apps = st.number_input("Number of Apps Installed", min_value=1, step=1)
data_usage = st.number_input("Data Usage (MB/day)", min_value=0, step=1)
device_model = st.selectbox("Device Model", ["Model A", "Model B", "Model C"])  # Adjust based on your dataset
operating_system = st.selectbox("Operating System", ["Android", "iOS"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert categorical inputs using label encoding
device_model_encoded = label_encoder.transform([device_model])[0]
operating_system_encoded = label_encoder.transform([operating_system])[0]
gender_encoded = label_encoder.transform([gender])[0]

# Normalize numerical inputs
scaled_inputs = scaler.transform([[app_usage_time, screen_on_time, battery_drain, num_apps, data_usage, age]])

# Combine all inputs into one array (include encoded categorical variables)
inputs = np.array([[*scaled_inputs[0], device_model_encoded, operating_system_encoded, gender_encoded]])

# Predict the behavior class
prediction = model.predict(inputs)

# Display prediction
st.write(f"Predicted User Behavior Class: {prediction[0]}")

# Optionally, show confidence scores
prediction_proba = model.predict_proba(inputs)
st.write(f"Prediction probabilities: {prediction_proba}")
