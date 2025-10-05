# SentAlyziz — IMDB Sentiment Analysis (ML & DL)

End-to-end sentiment analysis on IMDB movie reviews with **two complementary approaches**:

* **Traditional ML:** TF-IDF features + Logistic Regression & Linear SVM
* **Deep Learning:** LSTM over pre-trained **GloVe** embeddings (frozen & fine-tuned)




## Table of Contents

1. [Highlights](#highlights)
2. [What Was Done (End-to-End)](#what-was-done-end-to-end)
3. [Results (Summary)](#results-summary)
4. [Visualizations & Insights](#visualizations--insights)
5. [Repository Structure](#repository-structure)
6. [Getting Started](#getting-started)

   * [Environment (pyenv + venv)](#environment-pyenv--venv)
   * [Install Dependencies](#install-dependencies)
   * [IMDB Data Placement](#imdb-data-placement)
   * [Download GloVe (100d)](#download-glove-100d)
7. [How to Run (Notebooks)](#how-to-run-notebooks)
8. [Implementation Details](#implementation-details)

   * [Preprocessing](#preprocessing)
   * [ML Feature Extraction & Models](#ml-feature-extraction--models)
   * [DL Tokenization, Embeddings & Models](#dl-tokenization-embeddings--models)
9. [Artifacts & Reproducibility](#artifacts--reproducibility)
10. [Troubleshooting](#troubleshooting)
11. [Acknowledgments](#acknowledgments)
12. [Contributors](#contributors)


## Highlights

* **Deterministic** 80/10/10 stratified split with **no document overlap**.
* **Reusable preprocessing** with POS-aware lemmatization and stopword removal.
* **Rich EDA:** class balance, length distributions, punctuation density, n-grams, Zipf’s law.
* **Experiment tables & exported figures** (training curves, PR/ROC, confusion matrices).
* **Config-driven paths & seeds** (`src/config.py`) for consistent runs.
* **Professional structure**: separate notebooks for **ML** and **DL** experiments; reusable `src/` modules.

---

## What Was Done (End-to-End)

1. **Data ingest & deterministic splits**

   * Loaded an IMDB reviews CSV (binary labels 0/1).
   * Standardized columns, created a **reproducible** 80/10/10 **stratified** split.
   * Enforced **column contract** (`doc_id`, `text`, `label`) and **no cross-split overlap**.
   * Verified **class balance** is consistent across splits (pos ≈ 50%).

2. **EDA & preprocessing**

   * Previewed raw vs cleaned text with examples.
   * Implemented a **cleaning pipeline**: lowercasing → punctuation/number removal → tokenization → POS tagging → WordNet lemmatization → stopword removal → whitespace normalization.
   * Analyzed class balance, token length distributions (median ≈ 88), punctuation density, top uni/bi/tri-grams by class, and Zipf’s law.
   * Exported cleaned train/val/test CSVs and EDA figures.

3. **Traditional ML experiments**

   * Built **word TF-IDF (1–2-grams, 200k features)** and **char TF-IDF (3–5-grams, 100k)**; also **union** of the two.
   * Trained **Logistic Regression** and **Linear SVM** with a **C grid** {0.25, 0.5, 1, 2, 4, 8}.
   * Selected best by **validation macro-F1** (ties → accuracy).
   * Reported full metrics on **test**; exported confusion matrices, PR and ROC curves.

4. **Deep learning experiments**

   * Tokenized with Keras; padded sequences (`max_len=256`).
   * Loaded **GloVe-100d** and built an embedding matrix (OOV random init).
   * **Frozen GloVe**: trained 6 LSTM configs (units {64,128,256} × dropout {0.2,0.5}).
   * **Fine-tuned GloVe**: unfroze embeddings; two schedules (`lr=5e-4` and `lr=1e-4`) with EarlyStopping and best-epoch selection.
   * Exported training curves; saved model checkpoints.

5. **Comparison & conclusions**

   * TF-IDF + linear models reached the top scores.
   * LSTM + GloVe (especially fine-tuned) was competitive and improved over frozen embeddings, but slightly behind the ML baselines on this dataset.

---

## Results (Summary)

| Family | Model / Features                                       | Split    |  Accuracy | F1 (macro) |    PR-AUC |   ROC-AUC |
| ------ | ------------------------------------------------------ | -------- | --------: | ---------: | --------: | --------: |
| ML     | Logistic Regression — **word TF-IDF**                  | **Val**  |     0.908 |      0.908 |     0.967 |     0.967 |
| ML     | Logistic Regression — **word TF-IDF**                  | **Test** | **0.902** |  **0.902** | **0.965** | **0.966** |
| ML     | Linear SVM — **union (word+char)**                     | **Val**  | **0.911** |  **0.911** | **0.968** | **0.969** |
| ML     | Linear SVM — **union (word+char)**                     | **Test** |     0.901 |      0.901 |     0.965 |     0.966 |
| DL     | LSTM (GloVe 100d **frozen**), **u=64, d=0.5**          | **Val**  |     0.886 |      0.886 |     0.952 |     0.954 |
| DL     | LSTM (GloVe 100d **frozen**), **u=64, d=0.5**          | **Test** |     0.878 |      0.878 |     0.949 |     0.950 |
| DL     | LSTM (GloVe **fine-tuned**), **u=256, d=0.5, lr=5e-4** | **Val**  |     0.895 |      0.895 |     0.958 |     0.959 |
| DL     | LSTM (GloVe **fine-tuned**), **u=256, d=0.5, lr=5e-4** | **Test** |     0.892 |      0.892 |     0.958 |     0.959 |

**Takeaway:** strong n-gram TF-IDF features with linear models remain extremely competitive on IMDB. LSTM + GloVe benefits from fine-tuning and closes the gap, but stays slightly below the TF-IDF baselines.

---

## Visualizations & Insights

*All figures render in notebooks and export to `reports/figures/`.*

* **Class balance:** ~50/50 across splits (train pos≈50.19%, val pos≈50.18%, test pos≈50.19%).
* **Length stats (cleaned):** median ≈ **88** tokens; 95th percentile ≈ **306** tokens → motivates `max_len=256`.
* **Punctuation density:** skewed with a small set of high-exclamation reviews; minimal predictive power after cleaning.
* **Top n-grams:**

  * Negative: “bad movie”, “waste time”, “low budget”, …
  * Positive: “one best”, “must see”, “great movie”, …
* **Zipf curve:** strong long-tail frequency; a few very frequent tokens dominate.
* **ML curves:** PR/ROC demonstrate strong separability; confusion matrices show balanced FP/FN.
* **DL curves:**

  * Frozen GloVe converges in 2–4 epochs; larger LSTMs overfit faster than smaller ones.
  * Fine-tuning improves peaks; higher LR (5e-4) gains quickly but overfits earlier; lower LR (1e-4) is steadier.

---

## Repository Structure

```
.
├── data
│   ├── raw/                 # IMDB CSV (input)
│   ├── processed/           # cleaned train/val/test CSVs
│   ├── interim/             # tokenizer & temp artifacts  
│   └── external/            # GloVe embeddings            (gitignored)
├── models/                  # Saved .keras checkpoints    
├── notebooks/
│   ├── 00_Data_Ingest_and_Splits.ipynb
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 10_Experiments_ML.ipynb
│   └── 20_Experiments_DL.ipynb
├── reports/
│   ├── figures/             # exported plots
│   └── tables/              # CSV experiment tables
├── src/
│   ├── config.py            # paths, seed, constants
│   ├── data_io.py           # loading, splitting, saving
│   ├── preprocess.py        # text cleaning utilities
│   ├── vectorize.py         # TF-IDF vectorizers
│   ├── embeddings.py        # GloVe loader → embedding matrix
│   ├── models_ml.py         # LR / Linear SVM builders
│   ├── train_eval_ml.py     # ML training/eval helpers
│   ├── models_dl.py         # LSTM builders (frozen/FT)
│   ├── train_eval_dl.py     # DL training/eval helpers
│   ├── metrics.py           # metric helpers
│   ├── plots.py             # plotting & export utilities
│   └── utils.py
├── requirements.txt
└── README.md
```

Large/local artifacts (`data/external/`) are **intentionally gitignored**.


## Getting Started

### Environment (pyenv + venv)

```bash
# macOS (Homebrew) prerequisites
brew update
brew install pyenv openssl readline zlib bzip2

# init shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# install & activate Python 3.11.9
pyenv install 3.11.9
pyenv local 3.11.9
python -V
```

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### IMDB Data Placement

Place the IMDB reviews CSV in `data/raw/` with at least:

* `text` — the review content
* `label` — `0` (negative) or `1` (positive)

Run **`notebooks/00_Data_Ingest_and_Splits.ipynb`** to create:

* `data/processed/train.csv`, `val.csv`, `test.csv`
  The notebook verifies schema, **no overlaps**, and class balance consistency.

### Download GloVe (100d)

```bash
mkdir -p data/external
cd data/external
curl -L -C - -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip -j glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip
cd ../..
```

The DL notebook expects `data/external/glove.6B.100d.txt`.
If you change `EMB_DIM` in `src/config.py`, download the matching GloVe file (50d/200d/300d).

---

## How to Run (Notebooks)

1. **`00_Data_Ingest_and_Splits.ipynb`**

   * Create deterministic train/val/test splits with checks and printed stats.

2. **`01_EDA_and_Preprocessing.ipynb`**

   * EDA visuals (class balance, lengths, punctuation, n-grams, Zipf).
   * Run text cleaning and export cleaned splits.
   * Figures saved to `reports/figures/`.

3. **`10_Experiments_ML.ipynb`**

   * Fit TF-IDF vectorizers (word, char, union) on **train**.
   * Train **LR** & **Linear SVM** across **C grid**; select best on **val**.
   * Plot confusion matrices, PR, ROC; evaluate on **test**.
   * Export results table to `reports/tables/`.

4. **`20_Experiments_DL.ipynb`**

   * Tokenize/pad; build embedding matrix from **GloVe-100d**.
   * Train 6 **frozen** LSTMs; then **fine-tune** embeddings (two LRs).
   * EarlyStopping with best-epoch restore; save checkpoints to `models/`.
   * Export training curves and consolidated results table.

---

## Implementation Details

### Preprocessing

* Lowercasing → punctuation/number removal → tokenization
* POS tagging → **WordNet lemmatization** with POS → stopword removal
* Vectorizers/tokenizers fit on **train only**; applied to val/test (no leakage).
* Implemented in `src/preprocess.py` with `clean_text`, `clean_series`, and helpers.

### ML Feature Extraction & Models

* **Word TF-IDF**: `ngram_range=(1,2)`, `max_features=200_000`, `min_df=5`, `max_df=0.95`, `sublinear_tf=True`
* **Char TF-IDF**: `ngram_range=(3,5)`, `max_features=100_000`, `min_df=2`, `sublinear_tf=True`
* **Union** = horizontal stack of word + char TF-IDF matrices
* **Logistic Regression (liblinear)** & **Linear SVM**

  * Hyper-parameter grid: `C ∈ {0.25, 0.5, 1, 2, 4, 8}`
  * Selection by **validation macro-F1** (ties → accuracy)
* Metrics: **accuracy, macro-F1, PR-AUC, ROC-AUC** (val + test)

### DL Tokenization, Embeddings & Models

* Keras `Tokenizer` → integer sequences; `pad_sequences(maxlen=256)`
* **GloVe-100d** → embedding matrix; OOV tokens initialized randomly
* **Architectures:** `Embedding → LSTM(units, dropout) → Dense(1, sigmoid)`
* **Frozen**: 6 configs (units {64,128,256} × dropout {0.2,0.5})
* **Fine-tuned**: unfreeze embeddings, two schedules (`lr=5e-4` peaks faster; `lr=1e-4` steadier)
* EarlyStopping(patience=2, restore_best_weights=True), ModelCheckpoint per run
* Metrics: accuracy, macro-F1, PR-AUC, ROC-AUC (val + test); **training curves** exported

---

## Artifacts & Reproducibility

* **Config & Seeds:** centralized in `src/config.py` (e.g., `SEED`, paths).
* **Processed data:** `data/processed/` contains split CSVs (tracked).
* **Experiment tables:**

  * ML: `reports/tables/ml_results.csv`
  * DL: `reports/tables/dl_lstm_results.csv`
* **Figures:** exported to `reports/figures/` and visible inline in notebooks.
* **No leakage:** all transformers/tokenizers fitted on **train only**; val/test remain unseen.

---

## Troubleshooting

* **TensorFlow install (Apple Silicon):** use `tensorflow-macos==2.15.1` + `tensorflow-metal==1.1.0`.
* **GloVe not found:** ensure `data/external/glove.6B.100d.txt` exists; re-run the download commands.
* **OOM during DL:** reduce `BATCH_SIZE` (e.g., 64) or `MAX_LEN` (e.g., 200) in `src/config.py`.
* **Slow training:** close heavy apps; confirm you’re on Python **3.11.9** venv; limit notebook autosave.
* **GitHub push blocked:** large artifacts are gitignored; ensure you didn’t add  `data/external/` to Git.


## Acknowledgments

* IMDB movie reviews dataset
* GloVe (Pennington, Socher, Manning, 2014): Global Vectors for Word Representation

## Contributors
1. Christian Iradukunda Byiringiro
2. Armand Kayiranga
3. David Ubushakebwimana
4. Prince Rurangwa
