from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib

from app.config import MANIFEST_FILE
from app.dataset import FEATURE_NAMES, read_dataset
from app.torch_model import load_torch_bundle


def ensure_model_ready() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "registered model artifacts not found; run `make train` before serving or validating"
        )


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, object]:
    ensure_model_ready()
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_model(model_version: str | None = None):
    manifest = load_manifest()
    if model_version is None:
        return _load_registered_model(
            model_file=str(manifest["active_model_file"]),
            framework=str(manifest["active_model_framework"]),
        )
    if model_version not in manifest["available_models"]:
        raise KeyError(model_version)
    metadata = manifest["available_models"][model_version]
    return _load_registered_model(
        model_file=str(metadata["model_file"]),
        framework=str(metadata["framework"]),
    )


def _load_registered_model(*, model_file: str, framework: str):
    if framework == "torch":
        return load_torch_bundle(Path(model_file))
    return joblib.load(model_file)


def reload_model() -> None:
    load_manifest.cache_clear()
    load_model.cache_clear()


def available_models() -> dict[str, object]:
    manifest = load_manifest()
    return {
        "active_model_version": manifest["active_model_version"],
        "active_model_role": manifest["active_model_role"],
        "active_model_framework": manifest["active_model_framework"],
        "available_models": manifest["available_models"],
    }


def _prediction_payload(probability: float, model_version: str) -> dict[str, float | str]:
    label = "default_risk_high" if probability >= 0.5 else "default_risk_low"
    return {
        "model_version": model_version,
        "default_probability": round(probability, 6),
        "prediction": label,
    }


def predict(features: dict[str, float], model_version: str | None = None) -> dict[str, float | str]:
    manifest = load_manifest()
    resolved_version = model_version or str(manifest["active_model_version"])
    model = load_model(resolved_version)
    ordered_features = [[float(features[name]) for name in FEATURE_NAMES]]
    probability = float(model.predict_proba(ordered_features)[0][1])
    return _prediction_payload(probability, resolved_version)


def predict_many(
    records: list[dict[str, float]],
    model_version: str | None = None,
) -> list[dict[str, float | str]]:
    manifest = load_manifest()
    resolved_version = model_version or str(manifest["active_model_version"])
    model = load_model(resolved_version)
    ordered_features = [[float(record[name]) for name in FEATURE_NAMES] for record in records]
    probabilities = model.predict_proba(ordered_features)[:, 1]
    return [_prediction_payload(float(probability), resolved_version) for probability in probabilities]


def load_registered_batch(limit: int = 5) -> list[dict[str, float]]:
    manifest = load_manifest()
    dataset_rows = read_dataset(Path(manifest["dataset_file"]))
    selected_rows = dataset_rows[-limit:]
    return [{name: float(row[name]) for name in FEATURE_NAMES} for row in selected_rows]
