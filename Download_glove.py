#############################################
# downloard_glove.py: Fichier de vecteur pre-entrainé pour l'entrainement du modele
##############################################

import os
import shutil
import kagglehub

def download_glove():
    """
    Télécharge le fichier de vecteurs pré-entraînés GloVe (100 dimensions)
    depuis Kaggle et le copie dans le dossier `processed_data/`.

    Étapes :
    1. Crée le dossier `processed_data/` s'il n'existe pas.
    2. Vérifie si le fichier GloVe est déjà présent pour éviter un nouveau téléchargement.
    3. Télécharge le jeu de données depuis Kaggle via kagglehub.
    4. Copie le fichier téléchargé dans `processed_data/`.

    Exceptions :
        FileNotFoundError : Si le fichier GloVe n'est pas trouvé après le téléchargement.
    """

    # Chemin du dossier où se trouve ce script Python
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Dossier où sera stocké le fichier GloVe
    target_dir = os.path.join(base_dir, "processed_data")
    os.makedirs(target_dir, exist_ok=True)  # Création si inexistant

    # Nom du fichier GloVe attendu
    glove_filename = "glove.6B.100d.txt"
    glove_path = os.path.join(target_dir, glove_filename)

    # Étape 1 : Vérification si le fichier est déjà présent
    if os.path.exists(glove_path):
        print(f"{glove_filename} déjà présent dans processed_data/")
        return

    # Étape 2 : Téléchargement depuis Kaggle
    print("Téléchargement de GloVe depuis Kaggle...")
    dataset_path = kagglehub.dataset_download("danielwillgeorge/glove6b100dtxt")

    # Étape 3 : Vérification que le fichier a bien été téléchargé
    downloaded_glove_path = os.path.join(dataset_path, glove_filename)
    if not os.path.exists(downloaded_glove_path):
        raise FileNotFoundError(f"{glove_filename} introuvable après téléchargement.")

    # Étape 4 : Copie du fichier dans processed_data/
    shutil.copy(downloaded_glove_path, glove_path)
    print(f"{glove_filename} copié dans processed_data/")
    print(f"Fichier disponible ici : {glove_path}")


if __name__ == '__main__':
    # Exécution directe du script : télécharge GloVe
    download_glove()
