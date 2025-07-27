import streamlit as st
import pandas as pd
import joblib
 
model  = joblib.load('KNN_Heart.pkl')
scaler = joblib.load('scaler.pkl')
excepted_columns  = joblib.load('Columns.pkl')

st.title('Heart stroke Disease Prediction by Ashwani ❤️')

st.markdown(" provide the following details to predict heart stroke disease")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEx",['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0,1])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])
max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    input_data = {
        'age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_heart_rate,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'Slope_' + slope: 1
    }

    input_df = pd.DataFrame([input_data])
    for col in excepted_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[excepted_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts that you are at risk of heart stroke disease. Please consult a healthcare professional.")
    else:
        st.success("The model predicts that you are not at risk of heart stroke disease. Keep maintaining a healthy lifestyle!")