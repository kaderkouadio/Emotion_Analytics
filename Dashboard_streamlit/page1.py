import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json
import uuid


# --- Configuration de la page ---
st.set_page_config(
    page_title="À propos - Emotion_Analytics",
    page_icon="ℹ️",
    layout="centered"
)

# --- Titre principal ---
st.markdown(
    """
    <h1 style='text-align:center; color:#2E86C1;'>ℹ️ À propos</h1>
    <p style='text-align:center; font-size:1.1em;'>
    Découvrez le projet, ses objectifs et son auteur.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- Animation Lottie ---
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"
lottie_json = load_lottie_url(lottie_url)

if lottie_json:
    st_lottie(lottie_json, height=250, key="nlp_animation")
else:
    st.warning("⚠️ Impossible de charger l'animation Lottie.")


# --- Section : Description du projet ---
st.subheader("📌 Description du projet")
st.markdown(
    """
    **Emotion_Analytics** est une application NLP interactive permettant 
    de prédire automatiquement le **sentiment** (positif, neutre ou négatif) 
    de critiques de films.  

    Le projet comprend :
    - **Prétraitement des données textuelles** (nettoyage & normalisation)
    - **Téléchargement d'embeddings pré-entraînés** (GloVe)
    - **Entraînement d’un modèle de Deep Learning** converti en TFLite
    - **API REST avec FastAPI** pour exposer le modèle
    - **Dashboard Streamlit** pour l’analyse et la prédiction interactive
    """
)

# --- Section : Objectifs ---
st.subheader("🎯 Objectifs")
st.markdown(
    """
    - Offrir un outil simple et intuitif pour analyser les sentiments.  
    - Illustrer un **pipeline NLP complet**, de la donnée brute à la mise en production.  
    - Mettre en avant les performances d’un modèle léger (TFLite).  
    - Démontrer l'intégration de plusieurs outils (**TensorFlow, FastAPI, Streamlit**).  
    """
)

# --- Section : Auteur avec photo ---
st.subheader("👤 Auteur")
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://raw.githubusercontent.com/kaderkouadio/Emotion_Analytics/main/Dashboard_streamlit/Images/profil.jpg", width=120)
with col2:
    st.markdown(
    """
    <div style="text-align: center; line-height: 1.6;">
        <h2>👋 À propos de moi</h2>
        <p><strong>KOUADIO KADER</strong></p>
        <p>Économiste | Analyste Financier | Data Analyst | Développeur BI & Intelligence Artificielle<br>
        Passionné par l'analyse de données, le NLP et l'IA appliquée.</p>
        <p>
            <a href="https://www.linkedin.com/in/koukou-kader-kouadio-2a32371a4/" target="_blank">🔗 LinkedIn</a> |
            <a href="mailto:kkaderkouadio@gmail.com">📧 Email</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# --- Section : Compétences ---
st.subheader("🛠 Compétences")
st.markdown(
    """
    ### 📊 Finance & Économie
    - **Étude et analyse de projets** : évaluation de la rentabilité, analyse coûts-bénéfices, gestion des risques.
    - **Analyse financière et économique** : interprétation d’états financiers, ratios financiers, analyse macroéconomique et sectorielle.
    - **Modélisation financière et économique avancée**:  prévisions avancées, simulations de scénarios, valorisation d’actifs et modélisation de cash-flows.

    ### 📈 Data Analytics & Business Intelligence
    - **Outils** : Power BI, Tableau, Excel (avancé)  
    - **Langages** : SQL, DAX, M, Python (Pandas, NumPy, Matplotlib, Seaborn)  
    - **Concepts** : KPIs, reporting, ETL, analyse exploratoire, visualisation avancée  

    ### 🤖 Machine Learning & IA
    - **Types de modèles** : classification, régression, clustering  
    - **Algorithmes** : Régression Logistique, Random Forest, XGBoost, SVM  
    - **NLP & Analyse de sentiment**  : prétraitement, word embeddings (GloVe, Word2Vec), modèles de classification textuelle.
    - **Frameworks** : scikit-learn, TensorFlow, PyTorch, Streamlit  

    ### 🗄 Bases de données
    - **SGBD** : SQL Server, MySQL, PostgreSQL, SQLite , Informix, Teradata 
    - **Compétences** : conception, gestion et optimisation de requêtes  

    ### 💻 Développement
    - Développement d’applications avec **Streamlit** et **Flask**  
    - Création d’API REST avec **FastAPI**  
    - Automatisation de processus avec Python  
    """
)

# --- Section : Sources de données ---
st.subheader("🗂 Sources de données")
st.markdown(
    """
    - [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
    - [GloVe Embeddings (100d)](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)  
    """
)

# --- Section : Remerciements ---
st.subheader("🙏 Remerciements")
st.markdown(
    """
    Merci à la communauté **NLP & Open Source** pour les ressources,  
    et aux frameworks **Streamlit, FastAPI & TensorFlow** qui rendent ce projet possible.
    """
)
