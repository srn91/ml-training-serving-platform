from __future__ import annotations

import json
from functools import lru_cache

import joblib

from app.config import MANIFEST_FILE
from app.dataset import FEATURE_NAMES


def ensure_model_ready() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "registered model artifacts not found; run `make train` before serving or validating"
        )


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, str]:
    ensure_model_ready()
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model():
    manifest = load_manifest()
    return joblib.load(manifest["model_file"])


def reload_model() -> None:
    load_manifest.cache_clear()
    load_model.cache_clear()


def predict(features: dict[str, float]) -> dict[str, float | str]:
    model = load_model()
    ordered_features = [[float(features[name]) for name in FEATURE_NAMES]]
    probability = float(model.predict_proba(ordered_features)[0][1])
    label = "default_risk_high" if probability >= 0.5 else "default_risk_low"
    return {
        "model_version": load_manifest()["model_version"],
        "default_probability": round(probability, 6),
        "prediction": label,
    }
