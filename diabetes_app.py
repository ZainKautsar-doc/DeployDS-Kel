import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
# Jika pakai encoder (untuk categorical), load juga
# encoders = joblib.load('encoders.pkl')  # Kalau ada

# Title of the app
st.title("Prediksi Risiko Diabetes")

# Input fields
st.header("Masukkan Detail Pasien")
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glukosa", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Tekanan Darah (mmHg)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Tebal Lipatan Kulit (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.5)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4)
age = st.number_input("Usia", min_value=0, max_value=120, value=35)

# Prepare input data
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree_function],
    'Age': [age]
})

# Jika ada encoding, lakukan di sini
# (Tidak ada di dataset Pima Diabetes — semuanya numeric, jadi skip.)

# Scaling
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("Prediksi"):
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    prob_no_diabetes = probabilities[0] * 100
    prob_diabetes = probabilities[1] * 100

    st.subheader("Hasil Prediksi")
    if prob_diabetes < 30:
        st.success("Pasien diprediksi TIDAK Berisiko Diabetes.")
    elif 30 <= prob_diabetes <= 50:
        st.warning("Pasien memiliki Risiko Sedang terhadap Diabetes.")
    else:  # prob_diabetes > 50
        st.error("Pasien diprediksi Berisiko Diabetes.")

    st.write(f"Probabilitas Tidak Diabetes: {prob_no_diabetes:.2f}%")
    st.write(f"Probabilitas Diabetes: {prob_diabetes:.2f}%")

st.markdown("Catatan: Ini adalah alat prediksi berbasis model Random Forest. Konsultasikan dengan tenaga medis profesional untuk saran medis.")
