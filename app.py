import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Charger le modèle
model = joblib.load("model_diabetes.pkl")

st.set_page_config(page_title="Prédiction du diabète", layout="wide")

# Navigation dans la sidebar
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Accueil", "🔍 Prédiction", "📊 Analyse des résultats", "ℹ️ Explication de l’application"]
)

# === PAGE 1 : Accueil ===
if menu == "🏠 Accueil":
    st.title("🩺 Application de prédiction du diabète")
    st.markdown("""
    Bienvenue dans cette application.  
    Elle utilise un modèle de Machine Learning pour estimer le risque de diabète à partir de données médicales.  
    Naviguez dans le menu à gauche pour tester une prédiction, analyser les résultats ou comprendre l’objectif global de l’outil.
    """)

# === PAGE 2 : Prédiction ===
elif menu == "🔍 Prédiction":
    st.title("🔍 Faire une prédiction")

    st.sidebar.subheader("⚙️ Paramètres médicaux")
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

        # Graphe uniquement ici
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

# === PAGE 3 : Analyse des résultats ===
elif menu == "📊 Analyse des résultats":
    st.title("📊 Analyse des résultats")
    st.markdown("""
    Les résultats du graphe précédent permettent de comprendre le niveau de risque.  
    - **Barre verte (Pas de diabète)** : indique la probabilité estimée que la personne soit saine.  
    - **Barre rouge (Diabète)** : montre la probabilité que la personne soit diabétique selon les données fournies.  

    Cette analyse aide à visualiser l’équilibre entre les deux probabilités.  
    Une probabilité élevée du côté rouge est une alerte, tandis qu’une forte proportion de vert rassure sur l’absence de risque immédiat.  
    """)

# === PAGE 4 : Explication de l’application ===
elif menu == "ℹ️ Explication de l’application":
    st.title("ℹ️ Explication de l’application")
    st.markdown("""
    Cette application a été conçue dans un objectif pédagogique et médical.  
    Elle permet :  
    - d’illustrer comment des données médicales simples peuvent être utilisées dans un modèle de Machine Learning,  
    - de fournir une estimation rapide et visuelle du risque de diabète,  
    - d’explorer les possibilités d’outils interactifs pour aider au **dépistage précoce**.  

    ⚠️ Elle ne remplace pas un avis médical : elle doit être considérée comme un **outil de sensibilisation et de support**, non comme un diagnostic médical officiel.  
    """)

