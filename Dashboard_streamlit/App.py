import streamlit as st

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import re
import os
from typing import Optional
import tensorflow as tf   

# ======================
# Configuration de la page
# ======================
st.set_page_config(
    layout="wide",
    page_title="Emotion_Analytics",
    page_icon="📊",
)

# --- Configuration API & paths

MAX_LEN = 1000
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt"



# ======================
# Titre principal
# ======================
html_temp = """
    <div style="
        background: linear-gradient(90deg, #2c3e50, #3498db); 
        padding:15px; 
        border-radius:15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    ">
        <h1 style="color: white; text-align:center; font-size: 36px; margin:0;">
            📊 Dashboard Emotion_Analytics
        </h1>
    </div>
    <p style="
        font-size: 20px; 
        font-weight: bold; 
        text-align:center; 
        margin-top:15px;
    ">
        Analyse des sentiments 💡 | Pipeline NLP complet ⚙️ | Déploiement interactif 🚀
    </p>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# ======================
# Profil / Titre / Lien LinkedIn
# ======================
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height:100%;">
            <img src="https://raw.githubusercontent.com/kaderkouadio/Emotion_Analytics/main/Dashboard_streamlit/Images/profil.jpg"  
                 width="80" 
                 style="border-radius:50%; border:2px solid white; box-shadow:0px 2px 5px rgba(0,0,0,0.3);"/>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <h2 style='text-align: center; margin: 0;'>💡 Analyse et Prédiction des Sentiments</h2>
        <p style='text-align: center; margin-top: 0.2em; font-size:1.05em;'>
            NLP complet : prétraitement, embeddings GloVe, entraînement TFLite et mise en production.
        </p>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div style='display:flex; justify-content:flex-end; align-items:center; height:100%;'>
            <a href="https://www.linkedin.com/in/koukou-kader-kouadio-2a32371a4/" 
               target="_blank" 
               style='text-decoration: none; color: #0077b5; font-weight:bold; font-size:1.05em;'>
                👨‍💻 KOUADIO KADER
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write(" ")
st.write(" ")

# ======================
# Phase 1 : Pipeline NLP & Prétraitement
# ======================
st.markdown(
    """
    <h2 style='text-align: center; color: #3498db;'>Phase 1 : <strong>Pipeline NLP & Prétraitement</strong></h2>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])
with col1:
    st.image("https://raw.githubusercontent.com/kaderkouadio/Emotion_Analytics/main/Dashboard_streamlit/Images/images NLP.jpeg", width=600)

st.markdown(
    """
    <div style='text-align: justify; font-size: 16px; padding: 10px 30px;'>
        EmotionAI Analytics commence par la préparation des données :
        <ul>
            <li>Extraction des données IMDB</li>
            <li>Nettoyage et normalisation des textes (minuscule, suppression caractères spéciaux)</li>
            <li>Téléchargement des vecteurs pré-entraînés <strong>GloVe</strong></li>
            <li>Construction du tokenizer et séquences</li>
        </ul>
        Cette étape assure un pipeline robuste pour la suite de l'entraînement.
    </div>
    """,
    unsafe_allow_html=True
)

with col2:
    st.markdown(
    """
    <div style='
        border: 2px solid #3498db; 
        padding: 0.5em; 
        border-radius: 6px; 
        text-align: center; 
        margin-bottom: 0.8em;
    '>
        <h1 style='margin-bottom: 0.2em;'>⚙️ Prétraitement & Embeddings</h1>
    </div>
    <p style='text-align: center; font-size: 1.1em; margin-top: 0; margin-bottom: 0.5em;'>
        Un pipeline complet pour transformer les critiques brutes en données exploitables.
    </p>
    """,
    unsafe_allow_html=True
)

# ======================
# Phase 2 : Entraînement du Modèle
# ======================
st.markdown(
    """
    <h2 style='text-align: center; color: #3498db;'>Phase 2 : <strong>Entraînement du Modèle & Conversion TFLite</strong></h2>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])
with col1:
    st.image("https://raw.githubusercontent.com/kaderkouadio/Emotion_Analytics/main/Dashboard_streamlit/Images/TFLite.png", width=600)

st.markdown(
    """
    <div style='text-align: justify; font-size: 16px; padding: 10px 30px;'>
        Le cœur du projet est l'entraînement du modèle :
        <ul>
            <li>Construction d’un réseau de neurones (Embedding + LSTM/GRU)</li>
            <li>Optimisation et évaluation sur les données IMDB</li>
            <li>Conversion en <strong>TensorFlow Lite</strong> pour inférence rapide</li>
        </ul>
        Cette étape transforme les données en modèle exploitable en production.
    </div>
    """,
    unsafe_allow_html=True
)

with col2:
    st.markdown(
    """
    <div style='
        border: 3px solid #3498db; 
        border-radius: 8px; 
        padding: 12px; 
        text-align: center; 
        margin-bottom: 0.5em;
    '>
        <h1 style='margin: 0;'>🤖 Deep Learning</h1>
    </div>
    <p style='text-align: center; font-size: 1.1em; margin-top: 0.5em; margin-bottom: 0.5em;'>
        Entraînement et optimisation d’un modèle robuste de classification de sentiments.
    </p>
    """,
    unsafe_allow_html=True
)

# ======================
# Phase 3 : Déploiement & Application
# ======================
st.markdown(
    """
    <h2 style='text-align: center; color: #3498db;'>Phase 3 : <strong>Déploiement FastAPI & Application Streamlit</strong></h2>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])
with col1:
    st.image("https://raw.githubusercontent.com/kaderkouadio/Emotion_Analytics/main/Dashboard_streamlit/Images/FastAPI&Streamlit.png", width=600)

st.markdown(
    """
    <div style='text-align: justify; font-size: 16px; padding: 10px 30px;'>
        La dernière phase met le modèle à disposition :
        <ul>
            <li><strong>API FastAPI</strong> avec endpoints pour prédiction</li>
            <li><strong>Swagger UI</strong> pour tester les requêtes</li>
            <li><strong>Application Streamlit</strong> pour une interface utilisateur simple et interactive</li>
        </ul>
        Le projet est ainsi prêt pour une utilisation pratique et une intégration future.
    </div>
    """,
    unsafe_allow_html=True
)

with col2:
   st.markdown(
    """
    <div style='
        border: 3px solid #3498db; 
        border-radius: 8px; 
        padding: 12px; 
        text-align: center; 
        margin-bottom: 0.5em;
    '>
        <h1 style='margin: 0;'>🚀 Déploiement</h1>
    </div>
    <p style='text-align: center; font-size: 1.1em; margin-top: 0.5em; margin-bottom: 0.5em;'>
        API + Interface Streamlit : prédisez les sentiments de vos textes en direct.
    </p>
    <p style='text-align: center;'>
        <a href='https://emotion-analytics-jfqy.onrender.com' target='_blank' style='color: #1f77b4; text-decoration: none;'>
            🔗 Accéder a l'api
        </a>
    </p>
    """,
    unsafe_allow_html=True
)

# ======================
# Note d'information
# ======================
st.markdown(
    """
    <div style='
        margin-top: 30px;
        background-color: #e8f4fd;
        border-left: 5px solid #3498db;
        padding: 15px 20px;
        border-radius: 5px;
        font-size: 16px;
        color: #333;
    '>
        ℹ️ <strong>Note :</strong> Le premier chargement du modèle peut prendre quelques secondes car il initialise TensorFlow Lite et le tokenizer.
    </div>
    """,
    unsafe_allow_html=True
)
