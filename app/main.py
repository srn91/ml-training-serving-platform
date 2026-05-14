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
<style>
body{{margin:0;background:#f8fafc;color:#0f172a;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.5}}
main{{max-width:1080px;margin:0 auto;padding:56px 24px}}.hero{{background:linear-gradient(135deg,#111827,#7c3aed);color:white;border-radius:22px;padding:38px;box-shadow:0 24px 60px rgba(15,23,42,.18)}}
.eyebrow{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#ddd6fe;font-weight:700}}h1{{font-size:42px;line-height:1.05;margin:10px 0 14px}}.hero p{{font-size:17px;color:#ede9fe;max-width:780px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}}.card{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;box-shadow:0 10px 30px rgba(15,23,42,.06)}}
.metric{{font-size:25px;font-weight:800;color:#0f172a}}.label{{font-size:13px;color:#64748b;margin-top:3px}}.links{{display:flex;flex-wrap:wrap;gap:12px;margin-top:22px}}
a.button{{background:#0f172a;color:white;text-decoration:none;padding:11px 14px;border-radius:10px;font-weight:700}}a.secondary{{background:white;color:#0f172a;border:1px solid #cbd5e1}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}h1{{font-size:34px}}}}
</style></head>
<body><main>
<section class="hero"><div class="eyebrow">Model lifecycle platform</div><h1>ML Training Serving Platform</h1>
<p>Train-to-serve workflow with versioned model artifacts, active-model metadata, monitoring output, and prediction APIs.</p>
<div class="links"><a class="button" href="/model">Active model</a><a class="button secondary" href="/models">Model registry</a><a class="button secondary" href="/monitoring">Monitoring</a><a class="button secondary" href="/docs">API docs</a></div></section>
<section class="grid">
<div class="card"><div class="metric">{model_version}</div><div class="label">active model version</div></div>
<div class="card"><div class="metric">ready</div><div class="label">artifact status</div></div>
<div class="card"><div class="metric">&le;1e-6</div><div class="label">parity tolerance</div></div>
<div class="card"><div class="metric">multi</div><div class="label">version serving</div></div>
</section>
<section class="card"><p>The demo starts from persisted artifacts, exposes the active model and registered versions, and keeps the monitoring summary available from the same hosted service.</p></section>
</main></body></html>"""


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
