###########################################
# uvicorn api:app --reload: pour lancer l'api
###########################################
# -----------------------------
# Imports
# -----------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import re
import os
from typing import Optional
import tensorflow as tf   

# -----------------------------
# --- Configuration API & paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed_data")
TFLITE_PATH = os.path.join(PROCESSED_DIR, "Converted_model.tflite")
TOKENIZER_PATH = os.path.join(PROCESSED_DIR, "tokenizer.pickle")
MAX_LEN = 1000
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt"

api_description = f""" 
✨ Bienvenue sur l'API Emotion_Analysis 🧠

Cette API permet d'analyser les sentiments des critiques de films à l'aide d'un modèle de Deep Learning entraîné sur les données IMDB.  
Jeu de données GloVe : [GloVe 6B (100d)]({KAGGLE_DATASET_URL})

### Fonctionnalités principales :
- 🔍 Consulter les critiques et leurs sentiments prédits
- 🤖 Obtenir la prédiction de sentiment (positif, neutre ou négatif) pour un texte donné
- 📊 Accéder aux métriques d'évaluation du modèle
- ⚡ Supporte le traitement par lots pour plusieurs textes
"""

# -----------------------------
# --- Initialisation FastAPI
# -----------------------------
app = FastAPI(
    title="Emotion_Analysis API",
    description=api_description,
    version="1.0.0"
)

# -----------------------------
# --- Pydantic models
# -----------------------------
class PredictRequest(BaseModel):
    review: str = Field(..., min_length=1, max_length=5000)
    return_confidence: Optional[bool] = True

class PredictResponse(BaseModel):
    sentiment: str
    score: float

# -----------------------------
# --- Helpers
# -----------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return " ".join(text.split())

def map_sentiment(score: float) -> str:
    if score >= 0.65:
        return "positive"
    elif score <= 0.35:
        return "negative"
    else:
        return "neutral"

def preprocess_and_pad(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    return np.array(padded, dtype=np.float32)

# def clean_text(text: str) -> str:
#     text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
#     return " ".join(text.split())

# def map_sentiment(score: float) -> str:
#     return "positive" if score >= 0.5 else "negative"

# def preprocess_and_pad(text: str):
#     cleaned = clean_text(text)
#     seq = tokenizer.texts_to_sequences([cleaned])
#     padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
#     return np.array(padded, dtype=np.float32)

def run_tflite_inference(input_array: np.ndarray) -> float:
    interpreter.resize_tensor_input(input_details[0]['index'], input_array.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output[0][0])

# -----------------------------
# --- Startup event: load model & tokenizer
# -----------------------------
@app.on_event("startup")
def startup_event():
    global interpreter, input_details, output_details, tokenizer

    # Charger le tokenizer
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Charger le modèle TFLite
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        raise RuntimeError(f"Failed to load TFLite model: {e}")

# -----------------------------
# --- Endpoints
# -----------------------------
@app.get("/", summary="Vérifie l'état de l'API", tags=["Monitoring"])
async def root():
    return {"message": "Bienvenue sur l'API Emotion_Analysis. Statut : OK"}

@app.get("/health", summary="Vérifie la santé de l'API et la disponibilité du modèle", tags=["Monitoring"])
def health():
    return {"status": "ok", "model_loaded": os.path.exists(TFLITE_PATH) and os.path.exists(TOKENIZER_PATH)}

@app.get("/glove-info", tags=["Ressources"], summary="Informations sur le jeu de données GloVe")
def get_glove_info():
    return {
        "nom": "GloVe 6B (100d)",
        "source": KAGGLE_DATASET_URL,
        "description": "Vecteurs de mots pré-entraînés (100 dimensions) pour NLP"
    }

@app.post("/predict", response_model=PredictResponse, summary="Prédit le sentiment d'une critique de film", tags=["Prediction"])
def predict(req: PredictRequest):
    if not req.review or not req.review.strip():
        raise HTTPException(status_code=422, detail="Empty review text")
    try:
        X = preprocess_and_pad(req.review)
        score = run_tflite_inference(X)
        sentiment = map_sentiment(score)
        return PredictResponse(sentiment=sentiment, score=round(score, 4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
