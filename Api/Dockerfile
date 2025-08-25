#####################################
# Dockerfile
#####################################

# Utilise une image officielle Python 3.11.5 allégée
FROM python:3.11.5-slim

# Empêche Python de créer des fichiers .pyc (bytecode)
ENV PYTHONDONTWRITEBYTECODE=1

# Force l'affichage des logs Python en temps réel (sans buffer)
ENV PYTHONUNBUFFERED=1

# Définit le répertoire de travail dans le conteneur
WORKDIR /App

# Copie le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Met à jour pip puis installe les dépendances sans cache pour gagner de la place
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie tous les fichiers du dossier local vers le dossier de travail dans le conteneur
COPY . .

# Commande par défaut pour lancer le serveur Uvicorn sur le port 80 accessible depuis l'extérieur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]



