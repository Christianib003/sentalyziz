from __future__ import annotations
import matplotlib.pyplot as plt
from .config import REPORTS_FIG

def save_fig(name: str, tight: bool = True, formats=("png", "svg")):
    REPORTS_FIG.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    base = REPORTS_FIG / name
    for ext in formats:
        plt.savefig(f"{base}.{ext}", dpi=200)
