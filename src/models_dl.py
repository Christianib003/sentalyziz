from __future__ import annotations
from typing import Optional
from tensorflow import keras
from tensorflow.keras import layers

def make_lstm_classifier(
    vocab_size: int,
    max_len: int,
    embedding_dim: int,
    embedding_matrix: Optional["np.ndarray"] = None,
    lstm_units: int = 128,
    dropout_rate: float = 0.5,
    bidirectional: bool = True,
    embed_trainable: bool = False,
):
    inputs = keras.Input(shape=(max_len,), dtype="int32")
    if embedding_matrix is not None:
        emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim,
            weights=[embedding_matrix], input_length=max_len,
            mask_zero=True, trainable=embed_trainable, name="emb"
        )(inputs)
    else:
        emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim,
            input_length=max_len, mask_zero=True, name="emb"
        )(inputs)

    if bidirectional:
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(emb)
    else:
        x = layers.LSTM(lstm_units, return_sequences=False)(emb)

    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="lstm_clf")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
