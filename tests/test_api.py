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
    assert payload["model_version"] == "model-v1"
    assert 0.0 <= payload["default_probability"] <= 1.0


def test_health_and_model_endpoints_report_registered_version() -> None:
    client = _client()

    health_response = client.get("/health")
    model_response = client.get("/model")

    assert health_response.status_code == 200
    assert health_response.json()["model_version"] == "model-v1"
    assert model_response.status_code == 200
    assert model_response.json()["model_version"] == "model-v1"


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
