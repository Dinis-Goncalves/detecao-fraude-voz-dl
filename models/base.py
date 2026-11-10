from tensorflow.keras import layers, models

def create_model(max_len, vocab_size):
    """
    Base model: Embedding -> GlobalAveragePooling1D -> Dense -> Dropout -> Output
    Args:
        max_len (int): tamanho da sequencia máxima para o padding
        vocab_size (int): tamanho do vocabulario
    Returns:
        modelo Keras compilado e pronto a usar
    """
    base_model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=128),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    base_model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return base_model