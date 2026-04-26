from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from app.config import DATASET_CSV, MANIFEST_FILE, METRICS_FILE, MODEL_DIR, MODEL_FILE, SCHEMA_FILE
from app.dataset import FEATURE_NAMES, generate_rows, write_dataset


@dataclass(frozen=True)
class TrainingArtifacts:
    metrics: dict[str, float | int | str]


def _split_rows(rows: list[dict[str, float | int]]) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    split_index = int(len(rows) * 0.8)
    return rows[:split_index], rows[split_index:]


def _matrix(rows: list[dict[str, float | int]]) -> tuple[list[list[float]], list[int]]:
    features = [[float(row[name]) for name in FEATURE_NAMES] for row in rows]
    labels = [int(row["defaulted"]) for row in rows]
    return features, labels


def train_and_register() -> TrainingArtifacts:
    rows = generate_rows()
    write_dataset(rows, DATASET_CSV)
    train_rows, test_rows = _split_rows(rows)
    x_train, y_train = _matrix(train_rows)
    x_test, y_test = _matrix(test_rows)

    pipeline = RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=12,
        random_state=7,
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = pipeline.predict(x_test)

    metrics = {
        "model_version": "model-v1",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
                "model_version": "model-v1",
                "model_file": str(MODEL_FILE),
                "metrics_file": str(METRICS_FILE),
                "schema_file": str(SCHEMA_FILE),
                "dataset_file": str(DATASET_CSV),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainingArtifacts(metrics=metrics)
