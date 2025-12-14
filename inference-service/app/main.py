import os
import time
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator


# Model path (from inference-service/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "ml-training", "models", "xgboost_fraud_model.pkl"
)


class PredictionRequest(BaseModel):
    transaction_count_1m: int = Field(ge=0)
    avg_amount_1m: float = Field(ge=0)
    max_amount_1m: float = Field(gt=0)
    fraud_count_1m: int = Field(ge=0)


class PredictionResponse(BaseModel):
    is_fraud: Literal[0, 1]
    fraud_probability: float
    latency_ms: float
    model_version: str


app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Prometheus instrumentation (/metrics)
Instrumentator().instrument(app).expose(app)

# Load model at startup
model = joblib.load(MODEL_PATH)
MODEL_VERSION = "v1"


# Métricas of domain
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_version", "result"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
)

MODEL_VERSION_INFO = Gauge(
    "model_version_info",
    "Model version info",
    ["model_version"],
)
MODEL_VERSION_INFO.labels(model_version=MODEL_VERSION).set(1.0)

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Data drift score (0-1)",
)
DATA_DRIFT_SCORE.set(0.0)  


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.perf_counter()

    X = [[
        req.transaction_count_1m,
        req.avg_amount_1m,
        req.max_amount_1m,
        req.fraud_count_1m,
    ]]

    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)

    latency_sec = time.perf_counter() - start
    latency_ms = latency_sec * 1000.0

    # Metrcis of domain
    PREDICTIONS_TOTAL.labels(
        model_version=MODEL_VERSION,
        result=str(pred),
    ).inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_sec)

    return PredictionResponse(
        is_fraud=pred,
        fraud_probability=proba,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
    )
