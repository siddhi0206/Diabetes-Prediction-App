import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("log_reg_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below:")

# User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):

    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Likely Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Unlikely Diabetes (Probability: {probability:.2f})")