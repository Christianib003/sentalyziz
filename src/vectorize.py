from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple
import json

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

from .config import DATA_INTERIM, REPORTS_TAB

@dataclass
class VectMeta:
    kind: str  
    ngrams: str
    max_features: int
    min_df: int
    max_df: float
    sublinear_tf: bool
    vocab_size: Optional[int] = None

def _persist_vectorizer(vec: TfidfVectorizer, name: str, meta: VectMeta) -> None:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    dump(vec, DATA_INTERIM / f"{name}.joblib")
    meta.vocab_size = len(getattr(vec, "vocabulary_", {}) or {})
    REPORTS_TAB.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_TAB / f"{name}_meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)

def make_word_tfidf(
    train_text: Sequence[str],
    val_text: Sequence[str],
    test_text: Sequence[str],
    *,
    ngram_range=(1, 2),
    max_features=200_000,
    min_df=5,
    max_df=0.95,
    sublinear_tf=True,
    name="tfidf_word",
) -> Tuple[TfidfVectorizer, csr_matrix, csr_matrix, csr_matrix]:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=False, 
    )
    Xtr = vec.fit_transform(train_text)
    Xva = vec.transform(val_text)
    Xte = vec.transform(test_text)
    _persist_vectorizer(vec, name, VectMeta("word", str(ngram_range), max_features, min_df, max_df, sublinear_tf))
    return vec, Xtr, Xva, Xte

def make_char_tfidf(
    train_text: Sequence[str],
    val_text: Sequence[str],
    test_text: Sequence[str],
    *,
    ngram_range=(3, 5),
    max_features=100_000,
    min_df=2,
    max_df=1.0,
    sublinear_tf=True,
    name="tfidf_char",
) -> Tuple[TfidfVectorizer, csr_matrix, csr_matrix, csr_matrix]:
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=False,
    )
    Xtr = vec.fit_transform(train_text)
    Xva = vec.transform(val_text)
    Xte = vec.transform(test_text)
    _persist_vectorizer(vec, name, VectMeta("char", str(ngram_range), max_features, min_df, max_df, sublinear_tf))
    return vec, Xtr, Xva, Xte

def make_union(Xw_tr, Xw_va, Xw_te, Xc_tr, Xc_va, Xc_te):
    return (
        hstack([Xw_tr, Xc_tr]).tocsr(),
        hstack([Xw_va, Xc_va]).tocsr(),
        hstack([Xw_te, Xc_te]).tocsr(),
    )
