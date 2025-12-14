ML Production System
End-to-end MLOps pipeline demonstrating production-grade ML deployment and monitoring.

A complete system that goes from real-time data ingestion → feature engineering → model training → API serving → Kubernetes deployment → monitoring.

System Overview
This system is a production-grade MLOps pipeline that handles fraud detection for transactions in real-time. Data flows from Kafka topics through Spark Streaming for feature engineering, stores aggregated features in PostgreSQL, trains an XGBoost model (optionally tracked with MLflow), serves predictions via FastAPI, deploys to Kubernetes with auto-scaling, and is monitored with Prometheus/Grafana. The architecture prioritizes latency (<50ms predictions), scalability (10K predictions/sec), reliability (99.9% uptime), and maintainability (automated monitoring and clear retraining workflows).​

🎯 Use Case: Fraud Detection
This system predicts fraudulent transactions in real-time using:

Real-time data with Kafka streaming transaction data.

Features with Spark aggregating rolling window statistics.

Model with XGBoost trained on historical transactions.

Serving with FastAPI returning fraud probability in <50ms.

Deployment with Kubernetes with 3–10 replicas based on demand.

Monitoring with Prometheus and Grafana tracking latency, throughput, and drift.​

(Auto-retraining is designed conceptually but not wired with Airflow in this repo.)


🚀 Quick Start
Prerequisites

Docker & Docker Compose

Python 3.10+

Kubernetes (local cluster, e.g. OrbStack/minikube)

PostgreSQL client

Git​

1. Clone Repository
git clone https://github.com/ivaanblnc/ml-production-system.git
cd ml-production-system

2. Setup Local Environment (optional)

python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
bash scripts/setup-local.sh

3. Start Services

Option A: All-in-one with Docker Compose

docker-compose up -d
This starts (depending on your compose): Kafka & Zookeeper, PostgreSQL, Prometheus, Grafana, and the inference API.​

Option B: Individual Components

Terminal 1 – Data Pipeline:
d data-pipeline
docker-compose up
Terminal 2 – Kafka Producer:

bash
cd data-pipeline/kafka-producer
python producer.py
4. Train Model

bash
cd ml-training
python src/train.py
5. View Monitoring

Prometheus: http://localhost:9090

Grafana: http://localhost:3000 (admin/admin)

API metrics: http://localhost:8000/metrics​

📁 Project Structure

.github/workflows/: GitHub Actions CI/CD pipelines.

data-pipeline/: Kafka producer + Spark Streaming job for feature engineering (1-minute window aggregations).

ml-training/: Training scripts, notebooks, MLflow integration (optional) and model artifacts.

inference-service/ (o api/ en tu estructura): FastAPI server with prediction endpoints, Prometheus metrics, and health checks.

k8s/: K8s manifests for deployment, service, HPA scaling (3–10 replicas), ServiceMonitor, etc.

monitoring/: Prometheus configuration, Grafana dashboards, and data-drift logic.

scripts/: Helper scripts for setup, testing, deployment, and traffic generation.​
