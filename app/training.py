from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from app.config import (
    CHALLENGER_MODEL_FILE,
    CHAMPION_MODEL_FILE,
    COMPARISON_FILE,
    DATASET_CSV,
    MANIFEST_FILE,
    METRICS_FILE,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_VERSION,
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


def train_and_register() -> TrainingArtifacts:
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

    comparison = {
        "registry_version": MODEL_VERSION,
        "champion": champion,
        "challenger": challenger,
        "selected_model_role": selected["role"],
        "selected_model_version": selected["model_version"],
        "selection_reason": "higher holdout ROC AUC, then accuracy, then lower brier score",
    }
    COMPARISON_FILE.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

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
                "champion": champion,
                "challenger": challenger,
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
