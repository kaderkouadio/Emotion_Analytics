# 💬 Emotion_Analytics API

Bienvenue dans l’API **Emotion_Analytics** – une API RESTful développée avec **FastAPI** pour analyser le **sentiment** des textes via un modèle TensorFlow Lite (TFLite).  
Elle permet de prédire si un texte est **positif**, **neutre** ou **négatif**, avec un score de confiance.

---

## 🚀 Fonctionnalités principales

- 🔍 Analyse de sentiment sur n’importe quel texte soumis  
- 📊 Score de confiance associé à la prédiction  
- ⚡ Modèle optimisé au format **TensorFlow Lite** pour un déploiement rapide et léger  
- 🐳 Conteneur Docker prêt à l’emploi  
- 🌐 Documentation automatique via Swagger UI  
- ✔️ Endpoint `/health` pour monitoring et vérification de l’état

---

## ⚙️ Prérequis

- Python ≥ 3.11  
- Docker (optionnel, pour conteneuriser l’API)  
- Modèle pré-entraîné TensorFlow Lite (`Converted_model.tflite`)  
- Tokenizer sauvegardé (`tokenizer.pickle`)  
- Un client HTTP (ex : `httpx`, `requests`) pour tester l’API

---

## Installation rapide

# Cloner le dépôt
git clone https://github.com/kaderkouadio/Emotion_Analytics.git
cd Emotion_Analytics

# Installer les dépendances Python
pip install -r requirements.txt



#### ▶️ Lancer l’API
- En local (FastAPI + Uvicorn)
uvicorn api.main:app --host 0.0.0.0 --port 8000


## Auteur

Développé par [KADER KOUADIO](https://www.linkedin.com/in/koukou-kader-kouadio-2a32371a4/) en FastAPI  avec FastAPI et TensorFlow Lite.

---