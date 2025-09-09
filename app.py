import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("model_diabetes.pkl")

st.title("Prédiction du diabète avec Machine Learning")

# Champs de saisie
age = st.number_input("Âge", 1, 120, 30)
polyurie = st.selectbox("Polyurie", ["Yes", "No"])
polydipsie = st.selectbox("Polydipsie", ["Yes", "No"])
obesite = st.selectbox("Obésité", ["Yes", "No"])

# Encodage Yes/No
mapping = {"Yes":1, "No":0}
features = np.array([[age, mapping[polyurie], mapping[polydipsie], mapping[obesite]]])

if st.button("Prédire"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ Risque élevé de diabète")
    else:
        st.success("✅ Pas de risque détecté")
