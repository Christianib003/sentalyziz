from __future__ import annotations
from typing import Dict, List, Tuple
import json
import pandas as pd
from joblib import dump

from .config import MODELS_DIR, REPORTS_TAB
from .metrics import compute_metrics
from .models_ml import make_logreg, make_linear_svm, make_calibrated_svm
from .plots import save_fig

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def run_lr_grid(Xtr, ytr, Xva, yva, Xte, yte, C_grid, *, feature_name="word"):
    rows: List[Dict] = []
    best = None
    best_model = None
    best_val = -1.0

    for C in C_grid:
        model = make_logreg(C, seed=42)
        model.fit(Xtr, ytr)

        yva_pred = model.predict(Xva)
        yva_score = model.predict_proba(Xva)[:, 1]
        m = compute_metrics(yva, yva_pred, yva_score)
        m.update({"C": C, "features": feature_name})
        rows.append(m)

        if m["f1_macro"] > best_val:
            best_val = m["f1_macro"]
            best = {"C": C, **m}
            best_model = model

    # Save results table
    df = pd.DataFrame(rows)
    REPORTS_TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_TAB / "ml_lr_results.csv", index=False)

    # Evaluate best on test
    yte_pred = best_model.predict(Xte)
    yte_score = best_model.predict_proba(Xte)[:, 1]
    test_metrics = compute_metrics(yte, yte_pred, yte_score)

    # Persist model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_model, MODELS_DIR / f"ml_lr_best.joblib")

    return df, best, test_metrics, yte_pred, yte_score

def run_svm_grids(feature_sets, ytr, yva, yte, C_grid):
    rows: List[Dict] = []
    best = None
    best_model = None
    best_val = -1.0
    best_name = None

    for name, (Xtr, Xva, Xte) in feature_sets.items():
        for C in C_grid:
            base = make_linear_svm(C, seed=42)
            clf = make_calibrated_svm(base, cv=3)
            clf.fit(Xtr, ytr)

            yva_pred = clf.predict(Xva)
            yva_score = clf.predict_proba(Xva)[:, 1]
            m = compute_metrics(yva, yva_pred, yva_score)
            m.update({"C": C, "features": name})
            rows.append(m)

            if m["f1_macro"] > best_val:
                best_val = m["f1_macro"]
                best = {"features": name, "C": C, **m}
                best_model = clf
                best_name = name

    df = pd.DataFrame(rows)
    REPORTS_TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_TAB / "ml_svm_results.csv", index=False)

    # Evaluate on test
    Xte_best = feature_sets[best_name][2]
    yte_pred = best_model.predict(Xte_best)
    yte_score = best_model.predict_proba(Xte_best)[:, 1]
    test_metrics = compute_metrics(yte, yte_pred, yte_score)

    # Persist model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from joblib import dump
    dump(best_model, MODELS_DIR / f"ml_svm_best_{best_name}.joblib")

    return df, best, test_metrics, yte_pred, yte_score, best_name


def plot_confmat(y_true, y_pred, title: str, fname: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    save_fig(fname); plt.show()

def plot_pr(y_true, y_score, title: str, fname: str):
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(r, p)
    plt.title(f"{title} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    save_fig(fname); plt.show()

def plot_roc(y_true, y_score, title: str, fname: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(fpr, tpr)
    plt.title(f"{title} (AUC={auc:.3f})")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    save_fig(fname); plt.show()
