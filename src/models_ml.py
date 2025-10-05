from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def make_logreg(C: float, seed: int):
    return LogisticRegression(C=C, penalty="l2", solver="saga", max_iter=500, n_jobs=-1, random_state=seed)

def make_linear_svm(C: float, seed: int):
    return LinearSVC(C=C, loss="squared_hinge", dual=False, max_iter=5000, random_state=seed)

def make_calibrated_svm(base_svm: LinearSVC, cv: int = 3):
    return CalibratedClassifierCV(base_svm, method="sigmoid", cv=cv)
