import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🚭 Tobacco Mortality Prediction", layout="wide")
st.title("🚭 Tobacco Use & Mortality Risk Prediction Dashboard")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ----------------------------
# Input Sidebar
# ----------------------------
st.sidebar.header("🧾 Input Patient Details")

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex_encoded = 1 if sex == "Male" else 0

age_group_mapping = {
    "0–14": 0,
    "15–29": 1,
    "30–44": 2,
    "45–59": 3,
    "60+": 4
}
age_group = st.sidebar.selectbox("Age Group", list(age_group_mapping.keys()))
age_code = age_group_mapping[age_group]

log_admissions = st.sidebar.number_input("Log Admissions", value=0.0, format="%.2f")
admissions_normalized = st.sidebar.number_input("Admissions Normalized", value=0.0, format="%.2f")
year_numeric = st.sidebar.number_input("Year", value=2020)

if st.sidebar.button("Predict Risk 🧠"):
    features = np.array([sex_encoded, log_admissions, admissions_normalized, year_numeric, age_code]).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1] * 100

    with st.container():
        st.subheader("📊 Prediction Result")
        
        if prediction == 1:
            st.error(f"⚠️ High Mortality Risk — {probability:.2f}%")
        else:
            st.success(f"✅ Low Mortality Risk — {probability:.2f}%")

        st.write("### 🩺 Clinical Insights")
        if prediction == 1:
            st.write("""
            - 🚨 High Risk Detected  
            - Recommend medical monitoring  
            - Support tobacco cessation programs  
            - Consider screening for tobacco-related diseases  
            """)
        else:
            st.write("""
            - 🟢 Low Risk  
            - Maintain healthy lifestyle  
            - Continue routine health check-ups  
            """)

# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.caption("Developed by Archana — Tobacco Mortality ML Project 🚭")
