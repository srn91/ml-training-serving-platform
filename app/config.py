from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT_DIR / "generated"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_CSV = GENERATED_DIR / "credit_risk_dataset.csv"
MODEL_VERSION = "model-v1"
MODEL_DIR = ARTIFACTS_DIR / MODEL_VERSION
MODEL_FILE = MODEL_DIR / "active_model.artifact"
CHAMPION_MODEL_FILE = MODEL_DIR / "champion_model.joblib"
CHALLENGER_MODEL_FILE = MODEL_DIR / "challenger_model.joblib"
TORCH_MODEL_VERSION = f"{MODEL_VERSION}-torch"
TORCH_MODEL_FILE = MODEL_DIR / "torch_model.pt"
METRICS_FILE = MODEL_DIR / "metrics.json"
COMPARISON_FILE = MODEL_DIR / "comparison.json"
ROLLBACK_FILE = MODEL_DIR / "rollback.json"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
MANIFEST_FILE = MODEL_DIR / "manifest.json"
CALIBRATION_FILE = MODEL_DIR / "calibration.json"
DRIFT_BASELINE_FILE = MODEL_DIR / "drift_baseline.json"
MONITORING_SUMMARY_FILE = MODEL_DIR / "monitoring_summary.json"
