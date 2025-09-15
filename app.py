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

st.set_page_config(page_title="Prédiction du diabète", layout="wide")

# === Onglets principaux ===
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prédiction", "📊 Analyse exploratoire", "📂 Import CSV", "ℹ️ Explication"])

# ----------------- ONGLET 1 : PREDICTION -----------------
with tab1:
    st.header("🔍 Lancer la prédiction")

    # Sidebar pour les paramètres
    st.sidebar.header("⚙️ Paramètres médicaux")
    age = st.sidebar.slider("Quel est votre âge ?", 1, 120, 30)
    gender = st.sidebar.selectbox("Quel est votre sexe ?", ["Male", "Female"])
    polyuria = st.sidebar.selectbox("Avez-vous remarqué que vous urinez plus souvent que d’habitude ?)", ["Yes", "No"])
    polydipsia = st.sidebar.selectbox("Ressentez-vous une soif excessive même après avoir bu de l’eau ?", ["Yes", "No"])
    sudden_weight_loss = st.sidebar.selectbox("Avez-vous perdu du poids rapidement sans régime particulier ?", ["Yes", "No"])
    weakness = st.sidebar.selectbox("Ressentez-vous une fatigue ou une faiblesse inhabituelle au quotidien ?", ["Yes", "No"])
    polyphagia = st.sidebar.selectbox("Avez-vous souvent faim même après avoir mangé ?", ["Yes", "No"])
    genital_thrush = st.sidebar.selectbox("Souffrez-vous d’infections génitales répétées ?", ["Yes", "No"])
    visual_blurring = st.sidebar.selectbox("Votre vision devient-elle floue ou trouble par moments ?", ["Yes", "No"])
    itching = st.sidebar.selectbox("Avez-vous des démangeaisons fréquentes sur la peau ?", ["Yes", "No"])
    irritability = st.sidebar.selectbox("Ressentez-vous une irritabilité ou des changements d’humeur fréquents ?", ["Yes", "No"])
    delayed_healing = st.sidebar.selectbox("Vos plaies mettent-elles beaucoup de temps à cicatriser ?", ["Yes", "No"])
    partial_paresis = st.sidebar.selectbox("Avez-vous remarqué une faiblesse ou une perte de force dans certaines parties du corps ?", ["Yes", "No"])
    muscle_stiffness = st.sidebar.selectbox("Ressentez-vous une raideur ou une difficulté à bouger vos muscles ?", ["Yes", "No"])
    alopecia = st.sidebar.selectbox("Avez-vous observé une perte de cheveux inhabituelle récemment ?", ["Yes", "No"])
    obesity = st.sidebar.selectbox("Souffrez-vous d’obésité ou avez-vous un poids supérieur à la normale pour votre taille ?", ["Yes", "No"])

    mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
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

    if st.button("⚡ Lancer la prédiction"):
        prediction = model.predict(features)
        proba = model.predict_proba(features)[0][1]

        st.metric(label="Probabilité de diabète", value=f"{proba*100:.2f}%")

        if prediction[0] == 1:
            st.error("⚠️ Risque élevé de diabète détecté.")
        else:
            st.success("✅ Aucun risque de diabète détecté.")

        # Graphe matplotlib
        labels = ["Pas de diabète", "Diabète"]
        values = [1 - proba, proba]
        colors = ["green", "red"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilité")
        ax.set_title("Résultat de la prédiction", fontsize=14, fontweight="bold", pad=20)

        st.pyplot(fig)


# ----------------- ONGLET 2 : ANALYSE EXPLORATOIRE -----------------
with tab2:
    st.header("📊 Analyse exploratoire des données saisies")
    st.markdown("Voici un aperçu et une analyse simple des données entrées par l’utilisateur.")

    input_data = {
        "Age": age,
        "Gender": gender,
        "Polyurie": polyuria,
        "Polydipsie": polydipsia,
        "Perte de poids": sudden_weight_loss,
        "Faiblesse": weakness,
        "Polyphagie": polyphagia,
        "Mycose génitale": genital_thrush,
        "Vision trouble": visual_blurring,
        "Démangeaisons": itching,
        "Irritabilité": irritability,
        "Cicatrisation retardée": delayed_healing,
        "Paresie partielle": partial_paresis,
        "Raideur musculaire": muscle_stiffness,
        "Alopécie": alopecia,
        "Obésité": obesity
    }

    df_input = pd.DataFrame([input_data])
    st.dataframe(df_input)

    st.write("🔎 Cette table reprend toutes les caractéristiques fournies. Elle permet de vérifier la cohérence des données saisies avant la prédiction.")


# ----------------- ONGLET 3 : IMPORT CSV -----------------
with tab3:
    st.header("📂 Import d’un fichier CSV")
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Fichier bien chargé")
        st.write("Aperçu des données :")
        st.dataframe(df.head())

        st.write("Description statistique :")
        st.write(df.describe())


# ----------------- ONGLET 4 : EXPLICATION -----------------
with tab4:
    st.header("ℹ️ Explication de l’application")
    st.markdown("""
    Cette application a été développée pour montrer comment des données médicales peuvent être utilisées pour **prédire le risque de diabète** grâce au Machine Learning.  

    - **Onglet 1 : Prédiction** → permet à l’utilisateur d’entrer ses informations et d’obtenir une estimation du risque.  
    - **Onglet 2 : Analyse exploratoire** → permet de visualiser et analyser les données saisies.  
    - **Onglet 3 : Import CSV** → permet d’analyser un fichier de données complet (utile pour les médecins ou chercheurs).  
    - **Onglet 4 : Explication** → décrit l’objectif pédagogique et médical de l’application.  

    ⚠️ Remarque : cette application n’est pas un diagnostic médical officiel mais un outil de **sensibilisation** et de **support à la décision**.
    """)
