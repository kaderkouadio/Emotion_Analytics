
#############################################
# Model_train.py: Classe pour entraîner un modèle de classification de sentiments sur des critiques de films (Movie Review Sentiment Analysis).
##############################################
import tensorflow as tf
import numpy as np
import os
import pickle
import fire


class ReviewSentiment:
    """
    Classe pour entraîner un modèle de classification de sentiments
    sur des critiques de films (Movie Review Sentiment Analysis).

    Utilise :
    - Données prétraitées sauvegardées sous forme de fichiers .npy
    - Tokenizer sauvegardé en pickle
    - Embeddings GloVe pré-entraînés
    """

    def __init__(self, path, epochs):
        """
        Initialise les paramètres et charge les données.

        Args:
            path (str): Chemin vers le dossier `processed_data` contenant les fichiers traités.
            epochs (int): Nombre d'époques pour l'entraînement.
        """
        # Répertoire de base (où se trouve ce script)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Dossier de travail (processed_data/*)
        self.path = os.path.join(base_dir, path)
        self.epochs = epochs

        # Hyperparamètres
        self.batch_size = 250
        self.hidden_states = 100
        self.embedding_dim = 100
        self.learning_rate = 1e-4
        self.n_words = 50000 + 1
        self.sentence_length = 1000

        # Chargement des données
        self.X_train = np.load(os.path.join(self.path, "X_train.npy"))
        self.Y_train = np.load(os.path.join(self.path, "Y_train.npy")).reshape(-1, 1)
        self.X_val = np.load(os.path.join(self.path, "X_val.npy"))
        self.Y_val = np.load(os.path.join(self.path, "Y_val.npy")).reshape(-1, 1)
        self.X_test = np.load(os.path.join(self.path, "X_test.npy"))
        self.Y_test = np.load(os.path.join(self.path, "Y_test.npy")).reshape(-1, 1)

        print(f"Train set: {self.X_train.shape}, {self.Y_train.shape}")
        print(f"Validation set: {self.X_val.shape}, {self.Y_val.shape}")
        print(f"Test set: {self.X_test.shape}, {self.Y_test.shape}")
        print('Nombre de classes positives dans train:', np.sum(self.Y_train))
        print('Nombre de classes positives dans val:', np.sum(self.Y_val))

        # Chargement du tokenizer
        tokenizer_path = os.path.join(self.path, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Chargement des embeddings GloVe
        self.EMBEDDING_FILE = os.path.join(self.path, 'glove.6B.100d.txt')
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embedding_index = {}
        with open(self.EMBEDDING_FILE, encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = vector

        # Création de la matrice d'embedding
        word_index = tokenizer.word_index
        self.embedding_matrix = np.zeros((self.n_words, self.embedding_dim))
        for word, i in word_index.items():
            if i >= self.n_words:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def train(self):
        """
        Construit, entraîne et sauvegarde le modèle TensorFlow et sa version TFLite.
        """
        # Construction du modèle
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=self.n_words,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.sentence_length,
                trainable=True
            ),
            tf.keras.layers.LSTM(self.hidden_states),
            tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

        # Sauvegarde checkpoint
        checkpoint_path = os.path.join(self.path, "model_checkpoint.weights.h5")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        # Entraînement
        history = model.fit(
            self.X_train, self.Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.X_val, self.Y_val),
            callbacks=[cp_callback]
        )

        print("Entraînement terminé.")

        # Sauvegarde du modèle complet
        saved_model_path = os.path.join(self.path, 'saved_model.keras')
        model.save(saved_model_path)
        print(f"Modèle sauvegardé dans : {saved_model_path}")


        # Conversion en TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()

        tflite_model_path = os.path.join(self.path, "Converted_model.tflite")
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Modèle TFLite sauvegardé dans : {tflite_model_path}")

    def process_main(self):
        """
        Méthode principale pour lancer l'entraînement.
        """
        self.train()


if __name__ == '__main__':
    # Utilisation : python model_train.py ReviewSentiment --path processed_data --epochs 5 
    fire.Fire(ReviewSentiment)
