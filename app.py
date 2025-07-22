import streamlit as st
import numpy as np
import joblib

# Load model and other saved components
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Page title
st.title("Employee Salary Prediction App")
st.write("Enter details to predict if the person's income is >50K or <=50K")

# User inputs
age = st.number_input("Age", min_value=17, max_value=90, value=30)

workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
education = st.selectbox("Education", encoders['education'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
hours = st.slider("Hours per week", 1, 99, 40)

# When Predict button is clicked
if st.button("Predict Salary"):
    # Encode inputs
    workclass_encoded = encoders['workclass'].transform([workclass])[0]
    education_encoded = encoders['education'].transform([education])[0]
    occupation_encoded = encoders['occupation'].transform([occupation])[0]
    gender_encoded = encoders['gender'].transform([gender])[0]

    input_data = np.array([[age, workclass_encoded, education_encoded, occupation_encoded, gender_encoded, hours]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    result = target_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Income: {result}")
