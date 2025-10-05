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
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            pass 

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _map_pos(tag: str):
    if not tag:
        return wn.NOUN
    ch = tag[0].upper()
    return {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}.get(ch, wn.NOUN)

def clean_text(text: str) -> str:
    if text is None:
        return ""
    _ensure_nltk_data()

    s = str(text).lower() 
    s = _HTML_RE.sub(" ", s) 
    s = _URL_RE.sub(" ", s)  
    s = _PUNCT_RE.sub(" ", s)  
    tokens = s.split()                              
    if not tokens:
        return ""

    tagged = pos_tag(tokens)  
    lemmas = []
    for tok, tag in tagged:
        if len(tok) == 1 and tok not in _PUNCT_KEEP and not tok.isalnum():
            continue
        wn_pos = _map_pos(tag)
        lemma = _lemmatizer.lemmatize(tok, wn_pos)
        lemmas.append(lemma)

    sw = set(stopwords.words("english"))  
    keep = []
    for t in lemmas:
        if t in _PUNCT_KEEP:
            keep.append(t)
        elif t.isalpha() and t not in sw:
            keep.append(t)
        elif t.isdigit():
            keep.append(t)

    out = _WS_RE.sub(" ", " ".join(keep)).strip()
    return out

def clean_series(texts):
    import pandas as pd
    return texts.astype(str).map(clean_text)

def preview_cleaning(rows: Iterable[str], n: int = 5) -> List[Tuple[str, str]]:
    res = []
    for i, r in enumerate(rows):
        if i >= n:
            break
        res.append((r, clean_text(r)))
    return res
