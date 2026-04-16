# app.py

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved files
model = load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")
ct = joblib.load("encoder.pkl")

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("💳 Bank Customer Churn Prediction (ANN)")
st.write("Enter customer details:")

# Inputs
credit = st.number_input("Credit Score", 300, 900, 600)
geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 40)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 60000.0)
products = st.number_input("Number of Products", 1, 4, 2)
card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict"):

    try:
        # Encode gender same as training
        gender_val = 1 if gender == "Male" else 0

        # IMPORTANT: Keep Geography as STRING (not number)
        sample = np.array([[credit, geo, gender_val, age, tenure,
                            balance, products, card, active, salary]])

        # Apply same transformations
        sample = ct.transform(sample)
        sample = scaler.transform(sample)

        # Predict
        prediction = model.predict(sample)[0][0]

        st.subheader("Result:")

        if prediction > 0.5:
            st.error(f"❌ Customer will EXIT (Probability: {prediction:.2f})")
        else:
            st.success(f"✅ Customer will STAY (Probability: {prediction:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")