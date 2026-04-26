from fastapi.testclient import TestClient

from app.service import reload_model
from app.main import app
from app.training import train_and_register


def _client() -> TestClient:
    train_and_register()
    reload_model()
    return TestClient(app)


def test_predict_endpoint_returns_model_version_and_probability() -> None:
    client = _client()
    response = client.post(
        "/predict",
        json={
            "income_k": 72.0,
            "debt_to_income": 0.31,
            "credit_score": 690.0,
            "tenure_months": 36.0,
            "late_payments_12m": 1,
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["model_version"] in {"model-v1-champion", "model-v1-challenger"}
    assert 0.0 <= payload["default_probability"] <= 1.0


def test_health_and_model_endpoints_report_registered_version() -> None:
    client = _client()

    health_response = client.get("/health")
    model_response = client.get("/model")

    assert health_response.status_code == 200
    assert health_response.json()["active_model_role"] in {"champion", "challenger"}
    assert model_response.status_code == 200
    model_payload = model_response.json()
    assert model_payload["registry_version"] == "model-v1"
    assert model_payload["active_model_role"] in {"champion", "challenger"}
    assert model_payload["active_model_version"] in {"model-v1-champion", "model-v1-challenger"}
    assert "comparison_file" in model_payload
    assert "rollback_file" in model_payload


def test_batch_predict_endpoint_scores_multiple_records() -> None:
    client = _client()

    response = client.post(
        "/predict/batch",
        json={
            "records": [
                {
                    "income_k": 72.0,
                    "debt_to_income": 0.31,
                    "credit_score": 690.0,
                    "tenure_months": 36.0,
                    "late_payments_12m": 1,
                },
                {
                    "income_k": 48.0,
                    "debt_to_income": 0.52,
                    "credit_score": 640.0,
                    "tenure_months": 12.0,
                    "late_payments_12m": 3,
                },
            ]
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["model_version"] in {"model-v1-champion", "model-v1-challenger"}
    assert body["records_scored"] == 2
    assert len(body["predictions"]) == 2
    assert all(0.0 <= row["default_probability"] <= 1.0 for row in body["predictions"])


def test_predict_rejects_invalid_payload() -> None:
    client = _client()

    response = client.post(
        "/predict",
        json={
            "income_k": 72.0,
            "debt_to_income": 1.7,
            "credit_score": 690.0,
            "tenure_months": 36.0,
            "late_payments_12m": 1,
        },
    )

    assert response.status_code == 422
