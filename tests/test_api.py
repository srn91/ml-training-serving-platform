from fastapi.testclient import TestClient

from app.main import app


def test_predict_endpoint_returns_model_version_and_probability() -> None:
    client = TestClient(app)
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

