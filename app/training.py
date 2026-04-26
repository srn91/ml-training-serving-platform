from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from app.config import (
    CALIBRATION_FILE,
    CHALLENGER_MODEL_FILE,
    CHAMPION_MODEL_FILE,
    COMPARISON_FILE,
    DATASET_CSV,
    DRIFT_BASELINE_FILE,
    MANIFEST_FILE,
    METRICS_FILE,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_VERSION,
    MONITORING_SUMMARY_FILE,
    ROLLBACK_FILE,
    SCHEMA_FILE,
)
from app.dataset import FEATURE_NAMES, generate_rows, write_dataset


@dataclass(frozen=True)
class TrainingArtifacts:
    metrics: dict[str, object]


def _split_rows(
    rows: list[dict[str, float | int]],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    split_index = int(len(rows) * 0.8)
    return rows[:split_index], rows[split_index:]


def _matrix(rows: list[dict[str, float | int]]) -> tuple[list[list[float]], list[int]]:
    features = [[float(row[name]) for name in FEATURE_NAMES] for row in rows]
    labels = [int(row["defaulted"]) for row in rows]
    return features, labels


def _train_candidate(
    *,
    role: str,
    model_version: str,
    estimator,
    x_train: list[list[float]],
    y_train: list[int],
    x_test: list[list[float]],
    y_test: list[int],
    model_file: Path,
) -> dict[str, object]:
    model_file.parent.mkdir(parents=True, exist_ok=True)
    estimator.fit(x_train, y_train)
    probabilities = estimator.predict_proba(x_test)[:, 1]
    predictions = estimator.predict(x_test)
    metrics = {
        "role": role,
        "model_version": model_version,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
    }
    joblib.dump(estimator, model_file)
    return {
        "role": role,
        "model_version": model_version,
        "model_file": str(model_file),
        "metrics": metrics,
    }


def _selection_score(candidate: dict[str, object]) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    return (
        float(metrics["roc_auc"]),
        float(metrics["accuracy"]),
        -float(metrics["brier_score"]),
    )


def _calibration_summary(probabilities: np.ndarray, labels: list[int], bins: int = 5) -> dict[str, object]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket_rows: list[dict[str, float | int]] = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end == 1.0:
            mask = (probabilities >= start) & (probabilities <= end)
        else:
            mask = (probabilities >= start) & (probabilities < end)
        bucket_probs = probabilities[mask]
        bucket_labels = [label for label, keep in zip(labels, mask, strict=True) if keep]
        if len(bucket_probs) == 0:
            bucket_rows.append(
                {
                    "bin_start": round(float(start), 2),
                    "bin_end": round(float(end), 2),
                    "count": 0,
                    "mean_predicted_probability": 0.0,
                    "observed_default_rate": 0.0,
                }
            )
            continue
        bucket_rows.append(
            {
                "bin_start": round(float(start), 2),
                "bin_end": round(float(end), 2),
                "count": int(len(bucket_probs)),
                "mean_predicted_probability": round(float(bucket_probs.mean()), 4),
                "observed_default_rate": round(float(sum(bucket_labels) / len(bucket_labels)), 4),
            }
        )
    return {"bins": bucket_rows}


def _feature_baseline(rows: list[dict[str, float | int]]) -> dict[str, object]:
    baseline: dict[str, object] = {}
    for feature_name in FEATURE_NAMES:
        values = np.array([float(row[feature_name]) for row in rows], dtype=float)
        baseline[feature_name] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "p10": round(float(np.quantile(values, 0.10)), 4),
            "p50": round(float(np.quantile(values, 0.50)), 4),
            "p90": round(float(np.quantile(values, 0.90)), 4),
        }
    baseline["default_rate"] = round(float(sum(int(row["defaulted"]) for row in rows) / len(rows)), 4)
    return baseline


def train_and_register() -> TrainingArtifacts:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    write_dataset(rows, DATASET_CSV)
    train_rows, test_rows = _split_rows(rows)
    x_train, y_train = _matrix(train_rows)
    x_test, y_test = _matrix(test_rows)

    champion = _train_candidate(
        role="champion",
        model_version=f"{MODEL_VERSION}-champion",
        estimator=RandomForestClassifier(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=12,
            random_state=7,
        ),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model_file=CHAMPION_MODEL_FILE,
    )
    challenger = _train_candidate(
        role="challenger",
        model_version=f"{MODEL_VERSION}-challenger",
        estimator=GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            random_state=13,
        ),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model_file=CHALLENGER_MODEL_FILE,
    )

    selected = max((champion, challenger), key=_selection_score)
    alternate = challenger if selected is champion else champion
    selected_metrics = dict(selected["metrics"])
    selected_metrics["registry_version"] = MODEL_VERSION
    selected_metrics["selected_model_role"] = selected["role"]
    selected_metrics["selected_model_version"] = selected["model_version"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(joblib.load(selected["model_file"]), MODEL_FILE)

    champion_model = joblib.load(champion["model_file"])
    challenger_model = joblib.load(challenger["model_file"])
    champion_probs = champion_model.predict_proba(x_test)[:, 1]
    challenger_probs = challenger_model.predict_proba(x_test)[:, 1]

    comparison = {
        "registry_version": MODEL_VERSION,
        "champion": champion,
        "challenger": challenger,
        "selected_model_role": selected["role"],
        "selected_model_version": selected["model_version"],
        "selection_reason": "higher holdout ROC AUC, then accuracy, then lower brier score",
    }
    COMPARISON_FILE.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    calibration = {
        "registry_version": MODEL_VERSION,
        "champion": {
            "model_version": champion["model_version"],
            **_calibration_summary(champion_probs, y_test),
        },
        "challenger": {
            "model_version": challenger["model_version"],
            **_calibration_summary(challenger_probs, y_test),
        },
    }
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    drift_baseline = {
        "registry_version": MODEL_VERSION,
        "train_rows": len(train_rows),
        "features": _feature_baseline(train_rows),
    }
    DRIFT_BASELINE_FILE.write_text(json.dumps(drift_baseline, indent=2), encoding="utf-8")

    rollback = {
        "registry_version": MODEL_VERSION,
        "active_model_role": selected["role"],
        "active_model_version": selected["model_version"],
        "active_model_file": str(MODEL_FILE),
        "rollback_model_role": alternate["role"],
        "rollback_model_version": alternate["model_version"],
        "rollback_model_file": str(alternate["model_file"]),
        "reason": "alternate model remains packaged so the selected model can be rolled back without retraining",
    }
    ROLLBACK_FILE.write_text(json.dumps(rollback, indent=2), encoding="utf-8")

    METRICS_FILE.write_text(json.dumps(selected_metrics, indent=2), encoding="utf-8")
    SCHEMA_FILE.write_text(
        json.dumps(
            {
                "features": FEATURE_NAMES,
                "target": "defaulted",
                "prediction": "default_probability",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    MANIFEST_FILE.write_text(
        json.dumps(
            {
                "model_version": selected["model_version"],
                "registry_version": MODEL_VERSION,
                "active_model_role": selected["role"],
                "active_model_version": selected["model_version"],
                "active_model_file": str(MODEL_FILE),
                "comparison_file": str(COMPARISON_FILE),
                "rollback_file": str(ROLLBACK_FILE),
                "calibration_file": str(CALIBRATION_FILE),
                "drift_baseline_file": str(DRIFT_BASELINE_FILE),
                "monitoring_summary_file": str(MONITORING_SUMMARY_FILE),
                "champion": champion,
                "challenger": challenger,
                "available_models": {
                    champion["model_version"]: {
                        "role": champion["role"],
                        "model_file": str(CHAMPION_MODEL_FILE),
                    },
                    challenger["model_version"]: {
                        "role": challenger["role"],
                        "model_file": str(CHALLENGER_MODEL_FILE),
                    },
                },
                "metrics_file": str(METRICS_FILE),
                "schema_file": str(SCHEMA_FILE),
                "dataset_file": str(DATASET_CSV),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainingArtifacts(
        metrics={
            "registry_version": MODEL_VERSION,
            "selected_model_role": selected["role"],
            "selected_model_version": selected["model_version"],
            "champion": champion["metrics"],
            "challenger": challenger["metrics"],
            "selection_reason": "higher holdout ROC AUC, then accuracy, then lower brier score",
        }
    )
