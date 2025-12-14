import time
from typing import Literal
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from .db import SessionLocal, PredictionLog
from ..monitoring.monitoring import start_drift_monitor, DATA_DRIFT_SCORE


# Model path (from inference-service/)
MODEL_PATH = "/app/ml-training/models/xgboost_fraud_model.pkl"


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


class MonitoringResponse(BaseModel):
    model_version: str
    last_latency_ms: float
    data_drift_score: float
    total_requests: int



@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    start_drift_monitor(interval_seconds=60)
    yield

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)



# Prometheus instrumentation (/metrics)
Instrumentator().instrument(app).expose(app)

# Load model at startup
model = joblib.load(MODEL_PATH)
MODEL_VERSION = "v1"

# Domain metrics
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


LAST_PREDICTION_LATENCY_MS = 0.0
TOTAL_REQUESTS = 0


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/monitoring", response_model=MonitoringResponse)
def monitoring():
    return MonitoringResponse(
        model_version=MODEL_VERSION,
        last_latency_ms=LAST_PREDICTION_LATENCY_MS,
        data_drift_score=DATA_DRIFT_SCORE._value.get(),
        total_requests=TOTAL_REQUESTS,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    global LAST_PREDICTION_LATENCY_MS, TOTAL_REQUESTS

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

    # Domain metrics
    PREDICTIONS_TOTAL.labels(
        model_version=MODEL_VERSION,
        result=str(pred),
    ).inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_sec)

    # In-memory monitoring state
    LAST_PREDICTION_LATENCY_MS = latency_ms
    TOTAL_REQUESTS += 1

    db = SessionLocal()
    try:
        log = PredictionLog(
            transaction_count_1m=req.transaction_count_1m,
            avg_amount_1m=req.avg_amount_1m,
            max_amount_1m=req.max_amount_1m,
            fraud_count_1m=req.fraud_count_1m,
            is_fraud=bool(pred),
            fraud_probability=proba,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    return PredictionResponse(
        is_fraud=pred,
        fraud_probability=proba,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
    )
