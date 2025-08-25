
#############################################
# Extract.py: Extrait une archive tar.gz contenant les données IMDB
##############################################
import os
import tarfile

def extract_acrchive(archive_path: str, extract_to: str) -> None:
    """
    Extrait une archive tar.gz contenant les données IMDB si elle n'a pas déjà été extraite.

    Args:
        archive_path (str): Chemin complet vers l'archive .tar.gz à extraire.
        extract_to (str): Répertoire dans lequel extraire le contenu de l'archive.

    Returns:
        None
    """
    # Vérifie si le dossier "aclimdb" existe déjà dans le répertoire cible
    if not os.path.exists(os.path.join(extract_to, "aclimdb")):
        print(f"Extraction de {archive_path}...")

        # Ouvre l'archive en mode lecture gzip
        with tarfile.open(archive_path, "r:gz") as tar:
            # Extraction de tout le contenu dans le répertoire cible
            tar.extractall(path=extract_to)
            print("✅ Extraction terminée.")
    else:
        print("ℹ️ Les données sont déjà extraites.")

if __name__ == '__main__':
    # Détermine le répertoire de base du fichier courant
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construction des chemins vers l’archive et le dossier d’extraction
    archive_path = os.path.join(base_dir, "raw_data", "aclimdb_v1.tar.gz")
    extract_to = os.path.join(base_dir, "raw_data")

    # Appel de la fonction pour extraire l’archive
    extract_acrchive(archive_path, extract_to)
