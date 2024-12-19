import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Try to load the model
try:
    # Load the pre-trained model
    model = load_model(r"C:\Users\hassan\Desktop\diabetes prediction\diabeties_model_advance.keras.keras")
except Exception as e:
    # st.error(f"Error loading model: {e}")
    pass

# Set up Streamlit app
st.title("Diabetes Prediction Model")

# Input fields for the prediction
age = st.number_input("Age", min_value=0, value=30)
hypertension = st.selectbox("Hypertension", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.selectbox("Heart Disease", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
bmi = st.number_input("BMI", min_value=0.0, value=30.0)
hbA1c_level = st.number_input("HbA1c Level", min_value=0.0, value=6.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, value=120.0)

# Initialize the scaler
sc = StandardScaler()

# Load and prepare the dataset
df = pd.read_csv(r'C:\Users\hassan\Desktop\diabetes prediction\diabetes_prediction_dataset.csv')
df = df.drop_duplicates()
df.drop(['smoking_history', 'gender'], axis=1, inplace=True)
x, y = df.drop('diabetes', axis=1), df['diabetes']

# Fit the scaler to the training data
sc.fit(x)

# Predict button
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[age, hypertension, heart_disease, bmi, hbA1c_level, blood_glucose_level]])
    
    # Use the scaler to transform the input data
    input_data = sc.transform(input_data)

    # Make prediction
    predictions = model.predict(input_data)
    
    # Convert predictions to percentage confidence
    prediction = predictions[0][0] * 100
    st.write(f"Probability of having diabetes: {prediction:.2f}%")
