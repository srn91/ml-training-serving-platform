from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib

from app.config import MANIFEST_FILE
from app.dataset import FEATURE_NAMES, read_dataset


def ensure_model_ready() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "registered model artifacts not found; run `make train` before serving or validating"
        )


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, object]:
    ensure_model_ready()
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model():
    manifest = load_manifest()
    return joblib.load(manifest["active_model_file"])


def reload_model() -> None:
    load_manifest.cache_clear()
    load_model.cache_clear()


def _prediction_payload(probability: float) -> dict[str, float | str]:
    label = "default_risk_high" if probability >= 0.5 else "default_risk_low"
    return {
        "model_version": load_manifest()["active_model_version"],
        "default_probability": round(probability, 6),
        "prediction": label,
    }


def predict(features: dict[str, float]) -> dict[str, float | str]:
    model = load_model()
    ordered_features = [[float(features[name]) for name in FEATURE_NAMES]]
    probability = float(model.predict_proba(ordered_features)[0][1])
    return _prediction_payload(probability)


def predict_many(records: list[dict[str, float]]) -> list[dict[str, float | str]]:
    model = load_model()
    ordered_features = [[float(record[name]) for name in FEATURE_NAMES] for record in records]
    probabilities = model.predict_proba(ordered_features)[:, 1]
    return [_prediction_payload(float(probability)) for probability in probabilities]


def load_registered_batch(limit: int = 5) -> list[dict[str, float]]:
    manifest = load_manifest()
    dataset_rows = read_dataset(Path(manifest["dataset_file"]))
    selected_rows = dataset_rows[-limit:]
    return [{name: float(row[name]) for name in FEATURE_NAMES} for row in selected_rows]
