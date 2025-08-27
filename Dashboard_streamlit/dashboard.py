###############################################

import streamlit as st
from streamlit_lottie import st_lottie
import requests

# ---------------------------
# Configuration de la page
# ---------------------------
st.set_page_config(
    page_title="Emotion_Analytics",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------
# Navigation Multi-pages
# ---------------------------

# Définition des pages avec icônes
pages = {
    "🏠 Home": "App.py",                       
    "ℹ️ A propos": "page1.py",                  
    "🔎 Prédiction": "page2.py"                
}

# Barre latérale pour la navigation
st.sidebar.title("📌 Navigation")
selection = st.sidebar.radio("Aller à :", list(pages.keys()))

# Chargement de la page sélectionnée
page_file = pages[selection]
with open(page_file, "r", encoding="utf-8") as f:
    code = f.read()
    exec(code)
