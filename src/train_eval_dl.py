from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import DATA_INTERIM, MODELS_DIR, REPORTS_TAB
from .metrics import compute_metrics
from .plots import save_fig


def tokenize_and_pad(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    num_words: int = 60_000,
    oov_token: str = "<OOV>",
    max_len: int = 300,
):
    tok = Tokenizer(num_words=num_words, oov_token=oov_token, lower=False, split=" ")
    tok.fit_on_texts(train_texts)

    Xtr = pad_sequences(tok.texts_to_sequences(train_texts), maxlen=max_len, padding="post", truncating="post")
    Xva = pad_sequences(tok.texts_to_sequences(val_texts),   maxlen=max_len, padding="post", truncating="post")
    Xte = pad_sequences(tok.texts_to_sequences(test_texts),  maxlen=max_len, padding="post", truncating="post")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    with open(DATA_INTERIM / "tokenizer_config.json", "w") as f:
        f.write(tok.to_json())

    return tok, Xtr, Xva, Xte


def plot_history(history: keras.callbacks.History, title_prefix: str, fname_prefix: str):
    h = history.history
    # Accuracy
    plt.figure(figsize=(5,3.5))
    plt.plot(h["accuracy"], label="train_acc")
    plt.plot(h["val_accuracy"], label="val_acc")
    plt.title(f"{title_prefix} — Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()
    save_fig(f"{fname_prefix}_acc"); plt.show()

    # Loss
    plt.figure(figsize=(5,3.5))
    plt.plot(h["loss"], label="train_loss")
    plt.plot(h["val_loss"], label="val_loss")
    plt.title(f"{title_prefix} — Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    save_fig(f"{fname_prefix}_loss"); plt.show()

def evaluate_probs(model, X):
    scores = model.predict(X, verbose=0).ravel()
    preds = (scores >= 0.5).astype(int)
    return preds, scores

def train_lstm_experiment(
    model_maker,           
    Xtr, ytr, Xva, yva, Xte, yte,
    run_name: str,
    epochs: int = 12,
    batch_size: int = 128,
    patience: int = 2,
):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=str((MODELS_DIR / f"{run_name}.keras").resolve()),
            monitor="val_loss", save_best_only=True, save_weights_only=False)
    ]

    history = model_maker().fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    plot_history(history, title_prefix=run_name, fname_prefix=run_name)

    # Evaluate with best weights
    yva_pred, yva_score = evaluate_probs(history.model, Xva)
    val_metrics = compute_metrics(yva, yva_pred, yva_score)

    yte_pred, yte_score = evaluate_probs(history.model, Xte)
    test_metrics = compute_metrics(yte, yte_pred, yte_score)

    return {
        "history": history.history,
        "val": val_metrics,
        "test": test_metrics,
        "yte_pred": yte_pred,
        "yte_score": yte_score,
        "model_path": str(MODELS_DIR / f"{run_name}.keras"),
    } 

