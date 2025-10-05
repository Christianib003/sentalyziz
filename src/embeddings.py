from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_glove_embeddings(glove_txt_path: Path, embedding_dim: int) -> Dict[str, np.ndarray]:
    embeddings_index: Dict[str, np.ndarray] = {}
    with open(glove_txt_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading GloVe {embedding_dim}d"):
            parts = line.rstrip().split(" ")
            word = parts[0]
            coefs = np.asarray(parts[1:], dtype="float32")
            if coefs.shape[0] != embedding_dim:
                continue
            embeddings_index[word] = coefs
    return embeddings_index

def build_embedding_matrix(
    word_index: Dict[str, int],
    embeddings_index: Dict[str, np.ndarray],
    num_words: int,
    embedding_dim: int,
) -> np.ndarray:
    matrix = np.random.normal(scale=0.02, size=(num_words, embedding_dim)).astype("float32")
    for word, i in word_index.items():
        if i >= num_words: 
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            matrix[i] = vec
    return matrix
