from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT_DIR / "generated"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_CSV = GENERATED_DIR / "credit_risk_dataset.csv"
MODEL_VERSION = "model-v1"
MODEL_DIR = ARTIFACTS_DIR / MODEL_VERSION
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
MANIFEST_FILE = MODEL_DIR / "manifest.json"

