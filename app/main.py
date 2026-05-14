from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    manifest = load_manifest()
    model_version = manifest["active_model_version"]
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Training Serving Platform</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:860px;margin:48px auto;padding:0 24px;line-height:1.5;color:#111}}a{{color:#0645ad}}</style></head>
<body>
<h1>ML Training Serving Platform</h1>
<p>Train-to-serve workflow with versioned model artifacts, active-model metadata, monitoring output, and prediction APIs.</p>
<ul><li>Active model version: {model_version}</li><li>Artifacts ready: yes</li></ul>
<h2>Open endpoints</h2>
<ul>
<li><a href="/model">Active model manifest</a></li>
<li><a href="/models">Available models</a></li>
<li><a href="/monitoring">Monitoring summary</a></li>
<li><a href="/docs">API docs</a></li>
</ul>
</body></html>"""


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
