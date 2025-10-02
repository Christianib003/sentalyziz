from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
)

def compute_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}
    if y_score is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
            out["pr_auc"] = average_precision_score(y_true, y_score)
        except Exception:
            pass
    return out
