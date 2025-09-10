import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_page_config(
    page_title="Prédiction du diabète",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Prédiction du diabète")
st.write("Cette application utilise le Machine Learning pour estimer le risque de diabète à partir de vos données médicales.")
# Charger le modèle
model = joblib.load("model_diabetes.pkl")

st.title("Prédiction du diabète avec Machine Learning")

# Champs de saisie
age = st.sidebar.number_input("Âge", 1, 120, 30)
polyurie = st.sidebar.selectbox("Polyurie", ["Yes", "No"])
polydipsie = st.sidebar.selectbox("Polydipsie", ["Yes", "No"])
obesite = st.sidebar.selectbox("Obésité", ["Yes", "No"])

# Encodage Yes/No
mapping = {"Yes":1, "No":0}
features = np.array([[age, mapping[polyurie], mapping[polydipsie], mapping[obesite]]])

if st.button("Prédire"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ Risque élevé de diabète")
    else:
        st.success("✅ Pas de risque détecté")

# Section après la prédiction :

st.subheader("📊 Répartition simulée")
fig, ax = plt.subplots()
ax.bar(["Pas de diabète", "Diabète"], [1-probas, probas])
st.pyplot(fig)


#Bouton pour télécharger les résultats

if st.button("📥 Télécharger le rapport"):
    result = pd.DataFrame(features, columns=["Âge","Polyurie","Polydipsie","Obésité"])
    result["Probabilité diabète"] = [f"{probas*100:.2f}%"]
    result["Résultat"] = ["Risque élevé" if prediction[0] == 1 else "Pas de risque"]

    st.download_button(
        label="Télécharger en CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="rapport_prediction.csv",
        mime="text/csv"
    )
