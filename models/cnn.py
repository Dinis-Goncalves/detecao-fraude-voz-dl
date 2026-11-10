from tensorflow.keras import layers, models

def create_model(max_len, vocab_size):
    """
    CNN model: Embedding -> CNN -> Dense -> Dropout -> Output
    Args:
        max_len (int): tamanho da sequencia máxima para o padding
        vocab_size (int): tamanho do vocabulario
    Returns:
        modelo Keras compilado e pronto a usar
    """
    kernel_size = min(5, max_len)
    cnn_model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, 128),
        layers.SpatialDropout1D(0.2),

        layers.Conv1D(filters=128, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.GlobalMaxPooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    cnn_model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return cnn_model