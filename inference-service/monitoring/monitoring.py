import threading
import time
from typing import List

from prometheus_client import Gauge

from ..app.db import SessionLocal, PredictionLog

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Data drift score (0-1)",
)


def load_reference_data():
    db = SessionLocal()
    try:
        rows = (
            db.query(PredictionLog)
            .order_by(PredictionLog.ts.asc())
            .limit(1000)
            .all()
        )
    finally:
        db.close()
    return rows


def load_recent_data(limit: int = 500) -> List[PredictionLog]:
    db = SessionLocal()
    try:
        rows = (
            db.query(PredictionLog)
            .order_by(PredictionLog.ts.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()
    return rows


def compute_drift(recent_data, reference_data) -> float:
    if not recent_data or not reference_data:
        return 0.0

    ref_avg = sum(r.avg_amount_1m for r in reference_data) / len(reference_data)
    cur_avg = sum(r.avg_amount_1m for r in recent_data) / len(recent_data)

    if ref_avg == 0:
        return 0.0

    rel_change = abs(cur_avg - ref_avg) / ref_avg
    score = min(rel_change, 1.0)
    return float(score)


def drift_monitor_job(interval_seconds: int = 60):
    while True:
        reference_data = load_reference_data()
        recent_data = load_recent_data(limit=500)
        score = compute_drift(recent_data, reference_data)
        DATA_DRIFT_SCORE.set(score)
        time.sleep(interval_seconds)


def start_drift_monitor(interval_seconds: int = 60):
    t = threading.Thread(
        target=drift_monitor_job,
        args=(interval_seconds,),
        daemon=True,
    )
    t.start()
