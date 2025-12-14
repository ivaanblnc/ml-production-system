# inference-service/db.py

from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    Boolean,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# Postgres Settings (docker-compose)
DATABASE_URL = "postgresql://mluser:mlpass@postgres:5432/ml_data"

# Create engine and Session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)

    transaction_count_1m = Column(Integer, nullable=False)
    avg_amount_1m = Column(Float, nullable=False)
    max_amount_1m = Column(Float, nullable=False)
    fraud_count_1m = Column(Integer, nullable=False)

    is_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(Float, nullable=False)


# Create tables if they don't exist (runs on import of db.py)
Base.metadata.create_all(bind=engine)
