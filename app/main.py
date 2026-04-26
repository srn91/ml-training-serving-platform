from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.monitoring import build_monitoring_summary
from app.service import available_models, ensure_model_ready, load_manifest, predict, predict_many


class PredictionRequest(BaseModel):
    income_k: float = Field(ge=0.0)
    debt_to_income: float = Field(ge=0.0, le=1.0)
    credit_score: float = Field(ge=300.0, le=900.0)
    tenure_months: float = Field(ge=0.0)
    late_payments_12m: int = Field(ge=0, le=20)


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest] = Field(min_length=1, max_length=100)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_model_ready()
    yield


app = FastAPI(
    title="ML Training Serving Platform",
    version="0.1.0",
    description="Deterministic train-to-serve ML lifecycle demo with registry metadata and offline-online parity checks.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, object]:
    manifest = load_manifest()
    return {
        "project": "ml-training-serving-platform",
        "model_version": manifest["active_model_version"],
        "artifacts_ready": True,
    }


@app.get("/health")
def health() -> dict[str, object]:
    manifest = load_manifest()
    return {
        "status": "ok",
        "model_version": manifest["active_model_version"],
        "active_model_role": manifest["active_model_role"],
        "active_model_framework": manifest["active_model_framework"],
    }


@app.get("/model")
def model_info() -> dict[str, object]:
    return load_manifest()


@app.get("/models")
def models() -> dict[str, object]:
    return available_models()


@app.get("/monitoring")
def monitoring() -> dict[str, object]:
    return build_monitoring_summary()


@app.post("/predict")
def predict_route(request: PredictionRequest, version: str | None = None) -> dict[str, float | str]:
    return predict(request.model_dump(), model_version=version)


@app.post("/predict/batch")
def predict_batch_route(request: BatchPredictionRequest, version: str | None = None) -> dict[str, object]:
    records = [record.model_dump() for record in request.records]
    predictions = predict_many(records, model_version=version)
    return {
        "model_version": predictions[0]["model_version"] if predictions else load_manifest()["active_model_version"],
        "records_scored": len(predictions),
        "predictions": predictions,
    }
