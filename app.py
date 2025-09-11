import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Fonction pour ajouter une image de fond
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("");
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
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Diabetes_logo.svg/1200px-Diabetes_logo.svg.png",
    width=150
)
st.write("Cette application interactive utilise le Machine Learning pour estimer le risque de diabète à partir de données médicales.")

# Sidebar avec paramètres
age = st.sidebar.slider("Âge", 1, 120, 30)
gender = st.sidebar.selectbox("Sexe", ["Male", "Female"])
polyuria = st.sidebar.selectbox("Polyurie", ["Yes", "No"])
polydipsia = st.sidebar.selectbox("Polydipsie", ["Yes", "No"])
sudden_weight_loss = st.sidebar.selectbox("Perte de poids soudaine", ["Yes", "No"])
weakness = st.sidebar.selectbox("Faiblesse", ["Yes", "No"])
polyphagia = st.sidebar.selectbox("Polyphagie (faim excessive)", ["Yes", "No"])
genital_thrush = st.sidebar.selectbox("Mycose génitale", ["Yes", "No"])
visual_blurring = st.sidebar.selectbox("Vision trouble", ["Yes", "No"])
itching = st.sidebar.selectbox("Démangeaisons", ["Yes", "No"])
irritability = st.sidebar.selectbox("Irritabilité", ["Yes", "No"])
delayed_healing = st.sidebar.selectbox("Cicatrisation retardée", ["Yes", "No"])
partial_paresis = st.sidebar.selectbox("Paresie partielle", ["Yes", "No"])
muscle_stiffness = st.sidebar.selectbox("Raideur musculaire", ["Yes", "No"])
alopecia = st.sidebar.selectbox("Alopécie (perte de cheveux)", ["Yes", "No"])
obesity = st.sidebar.selectbox("Obésité", ["Yes", "No"])

# Encodage Yes/No → 1/0 et Male/Female → 1/0
mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

# Créer l'entrée pour le modèle (⚠️ respecter l'ordre des colonnes d'entraînement)
features = np.array([[
    age,
    mapping[gender],
    mapping[polyuria],
    mapping[polydipsia],
    mapping[sudden_weight_loss],
    mapping[weakness],
    mapping[polyphagia],
    mapping[genital_thrush],
    mapping[visual_blurring],
    mapping[itching],
    mapping[irritability],
    mapping[delayed_healing],
    mapping[partial_paresis],
    mapping[muscle_stiffness],
    mapping[alopecia],
    mapping[obesity]
]])
# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["📈 Prédiction", "📊 Analyse exploratoire", "📂 Import CSV", "ℹ️ Explication"])

# --------- Onglet 1 : Prédiction ---------
with tab1:
    st.subheader("Résultat de la prédiction")

    if st.button("🔍 Prédire"):
    prediction = model.predict(features)
    proba = model.predict_proba(features)[0][1]  # probabilité d'avoir le diabète

    st.metric(label="Probabilité de diabète", value=f"{proba*100:.2f}%")

    if prediction[0] == 1:
        st.error("⚠️ Risque élevé de diabète")
    else:
        st.success("✅ Pas de risque détecté")

         # Graphe interactif Plotly
    fig = go.Figure(go.Bar(
        x=["Pas de diabète", "Diabète"],
        y=[1-proba, proba],
        marker_color=["green", "red"],
        text=[f"{(1-proba)*100:.1f}%", f"{proba*100:.1f}%"],
        textposition="auto"
    ))

    fig.update_layout(
        title="Résultat interactif de la prédiction",
        yaxis=dict(title="Probabilité"),
        xaxis=dict(title="Classe"),
        bargap=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

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
