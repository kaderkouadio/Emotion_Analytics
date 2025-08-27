# 🎭 Emotion_Analytics  

## 📌 Description  
Emotion_Analytics est un projet d’analyse des sentiments qui permet de prédire l’émotion ou le sentiment associé à un texte (par exemple une critique de film).  
Le projet est structuré en **deux phases principales** :  

- **Phase 1 – Préparation & Entraînement des modèles**  
  - Préparation et nettoyage des données textuelles  
  - Vectorisation des textes (Word Embeddings, TF-IDF, etc.)  
  - Entraînement de modèles de NLP (Machine Learning / Deep Learning)  
  - Sauvegarde des modèles et du tokenizer dans le dossier `processed_data/`  

- **Phase 2 – Déploiement de l’application**  
  - API REST via **FastAPI** (hébergée sur Render)  
  - Interface utilisateur via **Streamlit** (dashboard interactif)  
  - Connexion Streamlit ↔️ API pour faire des prédictions en temps réel  

---

- **Une API REST (FastAPI)** pour la prédiction en temps réel.

- **Un tableau de bord Streamlit multipages** pour l’interaction utilisateur et la visualisation.

L’objectif est de permettre aux utilisateurs (scientifiques de données, analystes, décideurs, étudiants) :

- D’analyser automatiquement les émotions exprimées dans un texte.

- De tester des prédictions en direct depuis l’interface web.

- D’exploiter l’API pour intégrer le modèle dans d’autres systèmes.


## 🎯 Objectifs du projet

1. **Interface Utilisateur Interactive**

- Application web responsive avec Streamlit.
- Navigation multi-pages (Accueil, Visualisation, Prédiction, À propos).

2. **Visualisation Dynamique**

- Résultats clairs et graphiques simples pour illustrer les prédictions.
- Explication du processus de classification.

3. **Prédiction en Direct**

- Chargement du modèle NLP pré-entraîné.
- Saisie libre d’un texte par l’utilisateur.
- Retour immédiat de la prédiction avec score de confiance.

4. **Expérience Utilisateur**

- Animations Lottie pour dynamiser l’interface.
- Design cohérent et professionnel.


## 🛠 Fonctionnalités Principales

1. **Exploration / Démo des données** :

- Présentation des données utilisées pour l’entraînement.
- Aperçu interactif et descriptif.

2. **Prédiction en temps réel** :

- Interface simple où l’utilisateur tape son texte.
- Résultats instantanés (sentiment détecté + probabilité).

3. **Pages dédiées** :

- **Accueil** : présentation du projet et de ses objectifs.
- **Analyse** : aperçu des données et du modèle.
- **Prédiction** : simulation en direct.
- **À propos** : informations sur l’auteur et le projet.


## 📦 Livrables

- API **FastAPI** déployée en ligne sur Render.
- Application **Streamlit multipages** prête à l’emploi.
- Code Python structuré et documenté.
- Documentation complète (ce README).



## 🚀 Technologies Utilisées

-  **Langage** : Python 3.x
-  **Framework Web** : FastAPI (backend API), Streamlit (frontend)
-  **Machine Learning** : TensorFlow Lite, scikit-learn
-  **Visualisation** : Matplotlib, Plotly
-  **Animations** : streamlit-lottie
-  **Gestion des dépendances** : pip + requirements.txt
-  **Déploiement** : Render (API) + Streamlit


## 📌 Fonctionnalités

- 🧾 **API REST** : accès aux prédictions via endpoint /predict.

- 🎭 **Détection des émotions et sentiments** dans des textes.

- 📊 **Tableau de bord interactif** avec affichage en temps réel.

- 🎨 Interface moderne et animée avec **Lottie animations**.


## 🏗️ Structure du dépôt 
Emotion_Analytics/
│
├── processed_data/ # Modèles et tokenizer sauvegardés (Phase 1)
│ ├── model_sentiment.pkl
│ ├── tokenizer.pkl
│ └── ...
│
├── Dashboard_streamlit/ # Interface utilisateur Streamlit (Phase 2)
│ ├── dashboard.py
│ └── ...
│
├── api/ # API FastAPI
│ ├── main.py
│ └── ...
│
├── requirements.txt # Dépendances du projet
├── README.md # Documentation (ce fichier)
└── ...

## 🛠️ Installation

## 1️⃣ Cloner le dépôt

git clone https://github.com/kaderkouadio/Emotion_Analytics

cd Emotion_Analytics

## 2️⃣ Créer un environnement virtuel

python -m venv .venv
source .venv/bin/activate   ## macOS/Linux

.\venv\Scripts\activate     # Windows


## 3️⃣ Installer les dépendances

pip install -r requirements.txt

## 4️⃣ Lancer l’API

cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## 5️⃣ Lancer l’application Streamlit

cd Dashboard_streamlit
streamlit run app.py


## 🌐 Déploiement

🔗 API FastAPI : https://emotion-analytics-jfqy.onrender.com

🔗 Interface Streamlit : (lien à ajouter si déployée)


## 🔗 Me retrouver


[💼 LinkedIn](https://www.linkedin.com/in/koukou-kader-kouadio-2a32371a4/)

📧 [Email](mailto:kkaderkouadio@gmail.com)
