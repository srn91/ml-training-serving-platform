from app.training import train_and_register
from app.validation import validate_offline_online_parity


def test_training_metrics_are_strong_enough_for_demo() -> None:
    artifacts = train_and_register()

    assert artifacts.metrics["train_rows"] == 1920
    assert artifacts.metrics["test_rows"] == 480
    assert float(artifacts.metrics["accuracy"]) > 0.75
    assert float(artifacts.metrics["roc_auc"]) > 0.82


def test_offline_online_parity_stays_exact_for_local_model() -> None:
    summary = validate_offline_online_parity()

    assert summary.samples_checked == 25
    assert summary.max_probability_delta <= 1e-6
