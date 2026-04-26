from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import DRIFT_BASELINE_FILE, MONITORING_SUMMARY_FILE
from app.dataset import FEATURE_NAMES, read_dataset
from app.service import load_manifest, load_model


def _psi(reference_values: list[float], current_values: list[float], bins: int = 5) -> float:
    reference = np.array(reference_values, dtype=float)
    current = np.array(current_values, dtype=float)
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    psi = 0.0
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end == edges[-1]:
            ref_mask = (reference >= start) & (reference <= end)
            cur_mask = (current >= start) & (current <= end)
        else:
            ref_mask = (reference >= start) & (reference < end)
            cur_mask = (current >= start) & (current < end)
        ref_ratio = max(float(ref_mask.mean()), 1e-6)
        cur_ratio = max(float(cur_mask.mean()), 1e-6)
        psi += (cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)
    return round(float(psi), 4)


def _calibration_gap(bins: list[dict[str, float | int]]) -> float:
    populated = [row for row in bins if int(row["count"]) > 0]
    if not populated:
        return 0.0
    return round(
        max(
            abs(float(row["mean_predicted_probability"]) - float(row["observed_default_rate"]))
            for row in populated
        ),
        4,
    )


def build_monitoring_summary() -> dict[str, object]:
    manifest = load_manifest()
    dataset_rows = read_dataset(Path(str(manifest["dataset_file"])))
    baseline = json.loads(DRIFT_BASELINE_FILE.read_text(encoding="utf-8"))
    recent_rows = dataset_rows[-250:] if len(dataset_rows) >= 250 else dataset_rows

    feature_drift = {}
    for feature_name in FEATURE_NAMES:
        feature_drift[feature_name] = {
            "population_stability_index": _psi(
                [float(row[feature_name]) for row in dataset_rows[: int(len(dataset_rows) * 0.8)]],
                [float(row[feature_name]) for row in recent_rows],
            ),
            "recent_mean": round(float(np.mean([float(row[feature_name]) for row in recent_rows])), 4),
            "baseline_mean": baseline["features"][feature_name]["mean"],
        }

    calibration = json.loads(Path(str(manifest["calibration_file"])).read_text(encoding="utf-8"))
    models = {}
    for version, metadata in manifest["available_models"].items():
        model = load_model(version)
        probabilities = model.predict_proba([[float(row[name]) for name in FEATURE_NAMES] for row in recent_rows])[:, 1]
        models[version] = {
            "role": metadata["role"],
            "mean_recent_probability": round(float(probabilities.mean()), 4),
            "calibration_gap": _calibration_gap(calibration[metadata["role"]]["bins"]),
        }

    summary = {
        "registry_version": manifest["registry_version"],
        "active_model_version": manifest["active_model_version"],
        "recent_rows": len(recent_rows),
        "feature_drift": feature_drift,
        "models": models,
    }
    MONITORING_SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
