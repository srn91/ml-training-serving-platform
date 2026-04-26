from app.config import CALIBRATION_FILE, COMPARISON_FILE, DRIFT_BASELINE_FILE, MANIFEST_FILE, MODEL_FILE, MONITORING_SUMMARY_FILE, ROLLBACK_FILE
from app.training import train_and_register
from app.validation import validate_offline_online_parity


def test_training_metrics_are_strong_enough_for_demo() -> None:
    artifacts = train_and_register()

    assert artifacts.metrics["registry_version"] == "model-v1"
    assert artifacts.metrics["selected_model_role"] in {"champion", "challenger"}
    assert artifacts.metrics["champion"]["model_version"] == "model-v1-champion"
    assert artifacts.metrics["challenger"]["model_version"] == "model-v1-challenger"
    assert artifacts.metrics["champion"]["train_rows"] == 1920
    assert artifacts.metrics["champion"]["test_rows"] == 480
    assert float(artifacts.metrics["champion"]["accuracy"]) > 0.75
    assert float(artifacts.metrics["champion"]["roc_auc"]) > 0.82
    assert MANIFEST_FILE.exists()
    assert MODEL_FILE.exists()
    assert COMPARISON_FILE.exists()
    assert ROLLBACK_FILE.exists()
    assert CALIBRATION_FILE.exists()
    assert DRIFT_BASELINE_FILE.exists()


def test_offline_online_parity_stays_exact_for_local_model() -> None:
    summary = validate_offline_online_parity()

    assert summary.samples_checked == 25
    assert summary.max_probability_delta <= 1e-6
    assert "credit_score" in summary.monitoring_summary["feature_drift"]
    assert MONITORING_SUMMARY_FILE.exists()
