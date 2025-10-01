"""
Data I/O utilities for IMDb sentiment project (CSV-only).

- Load from local Kaggle CSV: data/raw/IMDB Dataset.csv
- Standardize to columns: text (str), label (0/1)
- Deduplicate exact (text,label) pairs
- Create deterministic doc_id = md5(text|label)
- Stratified 80/10/10 split with SEED
- Persist CSV splits and JSON stats

All paths & SEED come from src.config.
"""

from __future__ import annotations
import json, hashlib, random
from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import SEED, DATA_RAW, DATA_PROCESSED, REPORTS_TAB



def _set_seed(seed: int = SEED):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def _md5_text_label(text: str, label: int) -> str:
    s = (str(text).strip() + "|" + str(int(label))).encode("utf-8")
    return hashlib.md5(s).hexdigest()

def _counts(df: pd.DataFrame) -> Dict[str, float | int]:
    total = len(df)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return {
        "total": total,
        "pos": pos,
        "neg": neg,
        "pos_pct": round(100 * pos / total, 2) if total else 0.0,
        "neg_pct": round(100 * neg / total, 2) if total else 0.0,
    }


def load_imdb_from_csv(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load Kaggle CSV at data/raw/IMDB Dataset.csv by default.
    Returns DataFrame with columns: text (str), label (int 0/1).
    """
    path = DATA_RAW / "IMDB Dataset.csv" if csv_path is None else csv_path
    df = pd.read_csv(path)
    # Standardize columns
    if "review" in df.columns:
        df = df.rename(columns={"review": "text"})
    if "sentiment" in df.columns:
        df = df.rename(columns={"sentiment": "label"})
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must contain 'text' and 'label' (got {df.columns.tolist()})")
    # Map to 0/1 if string labels
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"negative": 0, "positive": 1})
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].copy()


def deduplicate_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicates on (text,label) and create doc_id.
    Returns (deduped_df with columns [doc_id,text,label], stats).
    """
    before = len(df)
    df = df[["text", "label"]].copy()
    df["doc_id"] = [_md5_text_label(t, y) for t, y in zip(df["text"], df["label"])]
    df = df.drop_duplicates(subset=["doc_id"]).copy()
    df = df[["doc_id", "text", "label"]]
    after = len(df)
    stats = {"before": before, "after": after, "removed": before - after}
    return df, stats

def stratified_80_10_10(df: pd.DataFrame, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic 80/10/10 stratified split. Expects columns: doc_id, text, label.
    """
    _set_seed(seed)
    y = df["label"].values

    # train (0.8) vs temp (0.2)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx_train, idx_temp = next(sss1.split(df, y))
    train = df.iloc[idx_train].reset_index(drop=True)
    temp  = df.iloc[idx_temp].reset_index(drop=True)

    # val (0.1 total) vs test (0.1 total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    y_temp = temp["label"].values
    idx_val, idx_test = next(sss2.split(temp, y_temp))
    val  = temp.iloc[idx_val].reset_index(drop=True)
    test = temp.iloc[idx_test].reset_index(drop=True)

    # overlap guards
    assert set(train.doc_id).isdisjoint(set(val.doc_id))
    assert set(train.doc_id).isdisjoint(set(test.doc_id))
    assert set(val.doc_id).isdisjoint(set(test.doc_id))

    return train, val, test

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, dedup_stats: dict | None = None):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_TAB.mkdir(parents=True, exist_ok=True)

    train.to_csv(DATA_PROCESSED / "train.csv", index=False)
    val.to_csv(  DATA_PROCESSED / "val.csv",   index=False)
    test.to_csv( DATA_PROCESSED / "test.csv",  index=False)

    stats = {"train": _counts(train), "val": _counts(val), "test": _counts(test)}
    with open(REPORTS_TAB / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    if dedup_stats is not None:
        with open(REPORTS_TAB / "dedup_stats.json", "w") as f:
            json.dump(dedup_stats, f, indent=2)


def ingest_and_split():
    """
    Orchestrate CSV load -> dedup -> split -> save (CSV-only workflow).
    """
    df = load_imdb_from_csv()
    df, dedup_stats = deduplicate_exact(df)
    train, val, test = stratified_80_10_10(df, seed=SEED)
    save_splits(train, val, test, dedup_stats=dedup_stats)
    return train, val, test
