#############################################
# preprocessed.py: Nettoie le texte brut en supprimant les caractères non alphabétiques et en normalisant en minuscules.
##############################################

import numpy as np
import pandas as pd
import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import fire

def text_clean(text):
    """
    Nettoie le texte brut en supprimant les caractères non alphabétiques 
    et en normalisant en minuscules.

    Args:
        text (str): Phrase ou texte à nettoyer.

    Returns:
        str: Texte nettoyé.
    """
    # Supprimer tout caractère qui n'est pas une lettre ou un espace
    letters = re.sub(r'[^a-zA-Z\s]', '', text)

    # Conversion en minuscules et séparation en mots
    words = letters.lower().split()

    # Reconstruction en phrase
    return ' '.join(words)


def process_train(input_path):
    """
    Lit les fichiers d'avis positifs et négatifs depuis les dossiers train/test,
    nettoie le texte et attribue les labels correspondants.

    Args:
        input_path (str): Chemin vers le dossier racine contenant les sous-dossiers train/ et test/.

    Returns:
        pandas.DataFrame: DataFrame contenant les colonnes :
            - Train_test_ind : "train" ou "test"
            - review : texte nettoyé
            - sentiment_label : 1 pour positif, 0 pour négatif
    """

    # Chemins vers les sous-dossiers
    review_paths = [
        (os.path.join(input_path, 'train', 'pos'), 1, 'train'),
        (os.path.join(input_path, 'train', 'neg'), 0, 'train'),
        (os.path.join(input_path, 'test', 'pos'), 1, 'test'),
        (os.path.join(input_path, 'test', 'neg'), 0, 'test'),
    ]

    reviews = []
    sentiment_labels = []
    train_test_labels = []

    # Parcours des dossiers
    for path, label, set_type in review_paths:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                reviews.append(text_clean(text))
                sentiment_labels.append(label)
                train_test_labels.append(set_type)

    # Création du DataFrame
    df = pd.DataFrame({
        'Train_test_ind': train_test_labels,
        'review': reviews,
        'sentiment_label': sentiment_labels
    })

    return df


def process_main(raw_data_path):
    """
    Point d'entrée principal :
    1. Charge et nettoie les données
    2. Tokenise et pad les séquences
    3. Sépare en ensembles train, validation et test
    4. Sauvegarde les fichiers traités dans processed_data/

    Args:
        raw_data_path (str): Chemin relatif vers les données brutes.
    """

    # Chemin absolu vers le dossier des données brutes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, raw_data_path)

    # Création du dossier de sortie
    output_path = os.path.join(base_dir, 'processed_data')
    os.makedirs(output_path, exist_ok=True)

    # Étape 1 : Chargement et nettoyage des données
    df = process_train(input_path)
    df.to_csv(os.path.join(output_path, 'processed_file.csv'), index=False)
    print(f"Nombre total d'enregistrements traités : {len(df)}")

    # Étape 2 : Tokenisation
    max_features = 50000  # Nombre max de mots dans le vocabulaire
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(df['review'].values)

    # Conversion en séquences numériques
    sequences = tokenizer.texts_to_sequences(df['review'].values)

    # Tronquer à 1000 tokens max par avis
    sequences = [seq[:1000] for seq in sequences]

    # Padding pour avoir des longueurs identiques
    X = pad_sequences(sequences)
    Y = df['sentiment_label'].values

    # Étape 3 : Mélange et découpage en train/val/test
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_end = int(0.7 * len(indices))
    val_end = train_end + int(0.15 * len(indices))

    X_train, Y_train = X[indices[:train_end]], Y[indices[:train_end]]
    X_val, Y_val = X[indices[train_end:val_end]], Y[indices[train_end:val_end]]
    X_test, Y_test = X[indices[val_end:]], Y[indices[val_end:]]

    # Étape 4 : Sauvegarde des fichiers numpy
    np.save(os.path.join(output_path, 'X_train.npy'), X_train)
    np.save(os.path.join(output_path, 'Y_train.npy'), Y_train)
    np.save(os.path.join(output_path, 'X_val.npy'), X_val)
    np.save(os.path.join(output_path, 'Y_val.npy'), Y_val)
    np.save(os.path.join(output_path, 'X_test.npy'), X_test)
    np.save(os.path.join(output_path, 'Y_test.npy'), Y_test)

    # Sauvegarde du tokenizer
    with open(os.path.join(output_path, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Données sauvegardées dans : {output_path}")


if __name__ == '__main__':
    # Utilisation avec la ligne de commande :
    # python preprocessing.py --raw-data-path raw_data/aclImdb
    fire.Fire(process_main)
