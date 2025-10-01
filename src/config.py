from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
REPORTS_FIG = BASE_DIR / "reports" / "figures"
REPORTS_TAB = BASE_DIR / "reports" / "tables"
MODELS_DIR = BASE_DIR / "models"

for p in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, REPORTS_FIG, REPORTS_TAB, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
