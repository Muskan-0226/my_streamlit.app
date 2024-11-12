import streamlit as st
import pandas as pd
import pickle

# Load the saved best model
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Define the Streamlit app
def main():
    st.title("User Behavior Classification")

    # Create input fields
    age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
    screen_time = st.number_input("Screen On Time (hours/day)", min_value=0.0, max_value=24.0, value=3.0, step=0.1)
    data_usage = st.number_input("Data Usage (MB/day)", min_value=0.0, value=500.0, step=10.0)
    apps_installed = st.number_input("Number of Apps Installed", min_value=0, max_value=1000, value=50, step=1)
    battery_drain = st.number_input("Battery Drain (mAh/day)", min_value=0.0, value=2000.0, step=10.0)
    device_model = st.selectbox("Device Model", ["iPhone", "Galaxy", "Pixel", "OnePlus"])
    os = st.selectbox("Operating System", ["iOS", "Android"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    # Create a sample input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Screen On Time (hours/day)': [screen_time],
        'Data Usage (MB/day)': [data_usage],
        'Number of Apps Installed': [apps_installed],
        'Battery Drain (mAh/day)': [battery_drain],
        'Device Model': [device_model],
        'Operating System': [os],
        'Gender': [gender]
    })

    # Preprocess the input data (encoding, scaling, etc.)
    # (Implement the same preprocessing steps as in the model.py file)

    # Make the prediction
    prediction = best_model.predict(input_data)[0]

    # Display the prediction
    st.subheader("Predicted User Behavior Class:")
    st.write(prediction)

if __name__ == '__main__':
    main()