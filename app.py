import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Fonction pour ajouter une image de fond
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1588776814546-0ce0c2e8577a");
             background-attachment: fixed;
             background-size: cover;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du diabète",
    page_icon="🩺",
    layout="wide"
)

# Charger le modèle
model = joblib.load("model_diabetes.pkl")

# Titre principal
st.title("🩺 Application de prédiction du diabète")
st.write("Cette application interactive utilise le Machine Learning pour estimer le risque de diabète à partir de données médicales.")

# Sidebar avec paramètres
age = st.sidebar.slider("Âge", 1, 120, 30)
gender = st.sidebar.selectbox("Sexe", ["Male", "Female"])
polyurie = st.sidebar.selectbox("Polyurie", ["Yes", "No"])
polydipsie = st.sidebar.selectbox("Polydipsie", ["Yes", "No"])
perte_poids = st.sidebar.selectbox("Perte de poids", ["Yes", "No"])
fatigue = st.sidebar.selectbox("Fatigue", ["Yes", "No"])
vision = st.sidebar.selectbox("Vision trouble", ["Yes", "No"])
itching = st.sidebar.selectbox("Démangeaisons", ["Yes", "No"])
irritabilite = st.sidebar.selectbox("Irritabilité", ["Yes", "No"])
cicatrisation = st.sidebar.selectbox("Retard de cicatrisation", ["Yes", "No"])
paresthesie = st.sidebar.selectbox("Paresthésie", ["Yes", "No"])
obesite = st.sidebar.selectbox("Obésité", ["Yes", "No"])
hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])

# Encodage Yes/No → 1/0 et Male/Female → 1/0
mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

# Créer l'entrée pour le modèle (⚠️ respecter l'ordre des colonnes d'entraînement)
features = np.array([[
    age,
    mapping[gender],
    mapping[polyurie],
    mapping[polydipsie],
    mapping[perte_poids],
    mapping[fatigue],
    mapping[vision],
    mapping[itching],
    mapping[irritabilite],
    mapping[cicatrisation],
    mapping[paresthesie],
    mapping[obesite],
    mapping[hypertension]
]])
# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["📈 Prédiction", "📊 Analyse exploratoire", "📂 Import CSV", "ℹ️ Explication"])

# --------- Onglet 1 : Prédiction ---------
with tab1:
    st.subheader("Résultat de la prédiction")

    if st.button("🔍 Lancer la prédiction"):
        prediction = model.predict(features)
        probas = model.predict_proba(features)[0][1]

        st.metric(label="Probabilité de diabète", value=f"{probas*100:.2f}%")

        # Messages conditionnels
        if probas > 0.7:
            st.warning("⚠️ Risque très élevé – consultez un médecin rapidement.")
        elif probas > 0.4:
            st.info("ℹ️ Risque modéré – un suivi médical est recommandé.")
        else:
            st.success("✅ Pas de risque détecté.")

        # Graphique visuel
        fig, ax = plt.subplots()
        ax.bar(["Pas de diabète", "Diabète"], [1-probas, probas], color=["green", "red"])
        st.pyplot(fig)

# --------- Onglet 2 : Analyse exploratoire ---------
with tab2:
    st.subheader("Analyse des données (exemple sur dataset)")
    try:
        df = pd.read_csv("diabetes_data.csv")  # place un dataset exemple
        fig = px.histogram(df, x="Age", color="class", nbins=20,
                           title="Répartition par âge et diabète")
        st.plotly_chart(fig)
    except:
        st.info("📌 Charge un fichier CSV dans l’onglet Import CSV pour analyser les données.")

# --------- Onglet 3 : Import CSV ---------
with tab3:
    st.subheader("📂 Importer vos données médicales")
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données :", user_data.head())

        try:
            predictions = model.predict(user_data)
            user_data["Résultat"] = ["Diabète" if p == 1 else "Non diabétique" for p in predictions]
            st.write("✅ Prédictions terminées :", user_data)

            # Téléchargement des résultats
            st.download_button(
                label="📥 Télécharger les prédictions",
                data=user_data.to_csv(index=False).encode("utf-8"),
                file_name="resultats_prediction.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# --------- Onglet 4 : Explication ---------
with tab4:
    st.subheader("ℹ️ Explication du modèle")
    st.write("""
    Ce modèle de Machine Learning a été entraîné à partir du dataset **Early Diabetes Risk Prediction**.
    Il prend en compte plusieurs variables médicales (âge, symptômes, habitudes) pour prédire si une personne
    présente un risque de diabète.  

    **Important :** Cette application est un outil pédagogique et ne remplace pas un diagnostic médical professionnel.
    """)
