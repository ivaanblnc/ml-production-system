FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Dependencies
COPY inference-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. API code
COPY inference-service ./inference-service

    # 3. Trained model (from ml-training/models/)
COPY ml-training/models/xgboost_fraud_model.pkl ./ml-training/models/xgboost_fraud_model.pkl

EXPOSE 8000

CMD ["uvicorn", "inference-service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
