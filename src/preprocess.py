"""
Deterministic text cleaning for IMDb sentiment.

Pipeline (document this in the report):
1) Lowercase
2) Strip HTML tags and URLs
3) Keep only ! and ? from punctuation; remove other punctuation/specials
4) Tokenize (whitespace split)
5) POS-aware lemmatization (WordNet)
6) Remove English stopwords (after lemmatization)
7) Collapse whitespace

Public API:
- clean_text(text: str) -> str
- clean_series(texts: pd.Series) -> pd.Series
- preview_cleaning(rows: Iterable[str], n=5) -> List[Tuple[orig, cleaned]]
"""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

_HTML_RE = re.compile(r"<.*?>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_PUNCT_KEEP = set("!?")
_PUNCT_RE = re.compile(r"[^\w\s!?]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

_lemmatizer = WordNetLemmatizer()

def _ensure_nltk_data():
    """Ensure required NLTK resources are available (idempotent)."""
    # stopwords
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    # wordnet + omw
    try:
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    # POS tagger (new name first, then legacy)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            pass  # fall through to legacy

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)

    # tokenizer data (punkt) – some NLTK builds need this for sentence/token ops
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _map_pos(tag: str):
    """Map Penn Treebank POS tags to WordNet POS for lemmatization."""
    if not tag:
        return wn.NOUN
    ch = tag[0].upper()
    return {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}.get(ch, wn.NOUN)

def clean_text(text: str) -> str:
    """Apply the deterministic cleaning pipeline to one string."""
    if text is None:
        return ""
    _ensure_nltk_data()

    s = str(text).lower()                           # 1) lowercase
    s = _HTML_RE.sub(" ", s)                        # 2) strip HTML
    s = _URL_RE.sub(" ", s)                         #    strip URLs
    s = _PUNCT_RE.sub(" ", s)                       # 3) drop punctuation except ! and ?
    tokens = s.split()                              # 4) whitespace tokenize
    if not tokens:
        return ""

    tagged = pos_tag(tokens)                        # 5) POS tag
    lemmas = []
    for tok, tag in tagged:
        if len(tok) == 1 and tok not in _PUNCT_KEEP and not tok.isalnum():
            continue
        wn_pos = _map_pos(tag)
        lemma = _lemmatizer.lemmatize(tok, wn_pos)
        lemmas.append(lemma)

    sw = set(stopwords.words("english"))            # 6) remove stopwords
    keep = []
    for t in lemmas:
        if t in _PUNCT_KEEP:
            keep.append(t)
        elif t.isalpha() and t not in sw:
            keep.append(t)
        elif t.isdigit():
            keep.append(t)

    out = _WS_RE.sub(" ", " ".join(keep)).strip()   # 7) collapse whitespace
    return out

def clean_series(texts):
    """Vectorized clean over a pandas Series; returns a new Series."""
    import pandas as pd
    return texts.astype(str).map(clean_text)

def preview_cleaning(rows: Iterable[str], n: int = 5) -> List[Tuple[str, str]]:
    """Return (original, cleaned) pairs for up to n rows (for documentation/report)."""
    res = []
    for i, r in enumerate(rows):
        if i >= n:
            break
        res.append((r, clean_text(r)))
    return res
