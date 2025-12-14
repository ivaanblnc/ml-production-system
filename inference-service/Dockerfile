FROM python:3.11-slim

# Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY app ./app

# Copy model from the repo (expected to be in the same build context)
# Adjust the path if your structure differs
COPY ../ml-training/models/xgboost_fraud_model.pkl ./ml-training/models/xgboost_fraud_model.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
