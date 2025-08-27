###############################################
import os
import sys
import importlib.util
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# ---------------------------

# ---------------------------
# Définition des pages avec icônes
pages = {
    "🏠 Home": "App.py",                       
    "👤 À propos": "page1.py",                 
    "🔎 Prédiction": "page2.py"                
}

# ---------------------------

st.sidebar.title("📌 Navigation")
selection = st.sidebar.radio("Aller à :", list(pages.keys()))

module_name = pages[selection]
module_path = os.path.join(BASE_DIR, f"{module_name}")

if os.path.exists(module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    st.error(f"❌ Fichier {module_path} introuvable")


