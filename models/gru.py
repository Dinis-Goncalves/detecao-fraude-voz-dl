from tensorflow.keras import layers, models

def create_model(max_len, vocab_size):
    """
    GRU model: Embedding -> Gru -> Dense -> Dropout -> Output
    Args:
        max_len (int): tamanho da sequencia máxima para o padding
        vocab_size (int): tamanho do vocabulario
    Returns:
        modelo Keras compilado e pronto a usar
    """
    gru_model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, 128),
        layers.SpatialDropout1D(0.2),
        layers.GRU(64),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    gru_model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return gru_model