from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.dataset import FEATURE_NAMES, read_dataset
from app.service import load_manifest, load_model, predict, reload_model


@dataclass(frozen=True)
class ValidationSummary:
    max_probability_delta: float
    samples_checked: int
    model_version: str


def validate_offline_online_parity() -> ValidationSummary:
    reload_model()
    manifest = load_manifest()
    rows = read_dataset(destination=Path(manifest["dataset_file"]))
    holdout = rows[-25:]

    model = load_model()
    max_delta = 0.0
    for row in holdout:
        features = {name: float(row[name]) for name in FEATURE_NAMES}
        direct_probability = float(
            model.predict_proba([[features[name] for name in FEATURE_NAMES]])[0][1]
        )
        service_probability = float(predict(features)["default_probability"])
        max_delta = max(max_delta, abs(direct_probability - service_probability))

    return ValidationSummary(
        max_probability_delta=round(max_delta, 8),
        samples_checked=len(holdout),
        model_version=str(manifest["model_version"]),
    )
