#############################################
# page3.py: Emotion_Analytics application streamlit pour prédire le sentiment à partir d'un modèle TFLite
##############################################

import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import re
import pickle
import os

# ------------------------------------------------------------
#  Chemins absolus vers le modèle et le tokenizer
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Dashboard_streamlit
PROJECT_DIR = os.path.dirname(BASE_DIR)               # dossier parent
MODEL_PATH = os.path.join(PROJECT_DIR, "processed_data", "Converted_model.tflite")
TOKENIZER_PATH = os.path.join(PROJECT_DIR, "processed_data", "tokenizer.pickle")

# ------------------------------------------------------------
#  Chargement du modèle et du tokenizer avec cache
# -----------------------------------------------------------

@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_tokenizer(path):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# ------------------------------------------------------------
#  Prétraitement du texte
# ------------------------------------------------------------
def preprocess_text(text, tokenizer, max_len=1000):
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    return padded.astype(np.float32)

# ------------------------------------------------------------
#  Prédiction du sentiment
# ------------------------------------------------------------
def predict_sentiment(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return float(output)

# ------------------------------------------------------------
#  Interface Streamlit
# ------------------------------------------------------------
st.markdown(
    """
    <div style='
        border: 3px solid #4CAF50; 
        border-radius: 10px; 
        padding: 12px; 
        text-align: center; 
        margin-bottom: 1em;
        background-color: #f9f9f9;
    '>
        <h1 style='margin: 0;'>🧠 Emotion_Analytics</h1>
    </div>
    <p style='text-align: center; font-size: 1.1em; margin-top: 0.5em;'>
        Écrivez votre critique de film ci-dessous et découvrez instantanément si le sentiment est positif ou négatif.
    </p>
    """,
    unsafe_allow_html=True
)

review_input = st.text_area("Votre critique :", height=150)

if st.button('Submit'):
    if review_input.strip() == '':
        st.warning("⚠️ Merci de saisir une critique avant de valider.")
    else:
        # Chargement du modèle et du tokenizer avec les chemins absolus
        interpreter = load_tflite_model(MODEL_PATH)
        tokenizer = load_tokenizer(TOKENIZER_PATH)

        # Prétraitement
        input_data = preprocess_text(review_input, tokenizer)

        # Prédiction
        score = predict_sentiment(interpreter, input_data)
        sentiment = "Positive 👍" if score >= 0.5 else "Negative 👎"

        # Affichage du résultat
        st.success(f"**Sentiment prédit :** {sentiment}")
        st.write(f"**Score de confiance :** {score:.4f}")
