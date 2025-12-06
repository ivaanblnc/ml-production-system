# ML Production System - Architecture Documentation

**Comprehensive technical architecture guide covering design decisions, data flows, and system components.**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline Architecture](#data-pipeline-architecture)
3. [ML Training Architecture](#ml-training-architecture)
4. [API & Serving Architecture](#api--serving-architecture)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Auto-retraining Pipeline](#auto-retraining-pipeline)
8. [Design Decisions](#design-decisions)
9. [Scalability Considerations](#scalability-considerations)
10. [Failure Modes & Mitigation](#failure-modes--mitigation)

---

## System Overview

This system is a production-grade MLOps pipeline that handles fraud detection for transactions in real-time. Data flows from Kafka topics through Spark Streaming for feature engineering, stores aggregated features in PostgreSQL, trains an XGBoost model with MLflow tracking, serves predictions via FastAPI, deploys to Kubernetes with auto-scaling, monitors with Prometheus/Grafana, and automatically retrains daily via Airflow when data drift is detected.

The architecture prioritizes latency (<50ms predictions), scalability (10K predictions/sec), reliability (99.9% uptime), and maintainability (automated retraining, comprehensive monitoring).

---

## Data Pipeline Architecture

### Kafka Topics Design

The system uses a single Kafka topic "transactions" with 12 partitions (sharded by customer_id % 12), replication factor of 3, and 7-day retention. Each message contains: transaction_id (UUID), customer_id, merchant_id, amount, currency, timestamp, card_last_4, is_fraud flag, country, and device_type.

Kafka was chosen because it provides fault-tolerance through replication, low-latency pub/sub messaging, support for multiple consumers simultaneously, event replay capability for debugging, and built-in backpressure handling. Alternatives like AWS Kinesis were rejected due to cost and vendor lock-in, RabbitMQ isn't designed for high-volume streaming, Redis lacks persistence, and while Pulsar is a good alternative, Kafka is the industry standard.

### Spark Streaming Architecture

The Spark Streaming job processes data in 1-second micro-batches with a 10-second watermark. Data flows: JSON parse → Watermark (10 seconds) → Aggregate by 1-minute window + customer_id → Calculate count(*) as transaction_count, avg(amount) as avg_amount, max(amount) as max_amount, count(is_fraud=1) as fraud_count, stddev(amount) as amount_std → Cache results → Output to PostgreSQL features table.

Configuration uses local[4] for local development or YARN/K8s for production. Batch duration is 1 second, watermark delay 10 seconds, output mode is Append (new features only). Checkpointing goes to HDFS in production (/tmp/spark-checkpoint locally). The 1-minute window was chosen because it provides good granularity for capturing customer behavior patterns, achieves <2 minutes end-to-end latency for fresh features, maintains manageable memory footprint for state, and balances accuracy vs. latency vs. cost.

Checkpointing strategy writes state to a checkpoint directory, enabling recovery from failures, preventing duplicate processing, and allowing exactly-once semantics. Spark configuration: Master is local[4] (development) or YARN/K8s (production), batch duration 1 second, watermark delay 10 seconds, output mode Append.

### Feature Store (PostgreSQL)

The PostgreSQL feature store contains multiple tables: transactions (stores raw incoming data with indexed customer_id and timestamp), features (aggregated 1-minute window features with customer_id and window_start indexed), training_features (materialized features joined with fraud labels for ML training), and predictions (log of all predictions for monitoring and drift detection).

PostgreSQL was selected because it provides ACID guarantees, fast indexed retrieval, familiar SQL interface, pgVector extension for future vector embeddings, and easy scaling via partitioning. Alternatives: Tecton is premium/enterprise only, open-source feature stores are immature, Redis works as cache only without persistence, DynamoDB has eventual consistency issues.

---

## ML Training Architecture

### Training Pipeline

The training process: (1) Load data from PostgreSQL with last 30 days of transactions, use 80/20 train/test split. (2) Preprocessing: handle missing values, scale features with StandardScaler, encode categorical variables. (3) Train model: XGBoost classifier with 5-fold cross-validation and hyperparameter tuning. (4) Evaluate: calculate AUC, Precision, Recall, feature importance, confusion matrix. (5) Track with MLflow: log params, metrics, and model artifacts. (6) Version with DVC: lock data snapshot, model weights, and parameters. (7) Register with MLflow: push to Model Registry in Staging stage with metadata tags.

### Model Selection: XGBoost

XGBoost was chosen over alternatives: accuracy (⭐⭐⭐⭐⭐), speed (⭐⭐⭐⭐), interpretability (⭐⭐⭐⭐), training time (⭐⭐⭐⭐), production-ready (⭐⭐⭐⭐⭐). Neural networks score better on accuracy/interpretability tradeoff but slower inference, need GPUs, black-box. Random Forest is simpler but less accurate. Logistic Regression is fastest but least accurate.

Hyperparameters: n_estimators=100 (trees in ensemble), max_depth=6 (prevents overfitting), learning_rate=0.1 (shrinkage factor), subsample=0.8 (row sampling), colsample_bytree=0.8 (column sampling), min_child_weight=1 (min samples in leaf), reg_lambda=1.0 (L2 regularization), reg_alpha=0.0 (L1 regularization).

### MLflow Experiment Tracking

Each training run creates an experiment with tracked params.json (hyperparameters), metrics/ folder (auc.json, precision.json, recall.json scores), and artifacts/ folder (model.pkl trained model, feature_importance.csv, confusion_matrix.png visualization, meta.yaml metadata).

MLflow was chosen for experiment comparison, model versioning, Model Registry capabilities, easy integration, and being open-source. It allows viewing all experiments, comparing metrics across runs, versioning models (v1, v2, v3), and staging transitions (Dev → Staging → Production).

---

## API & Serving Architecture

### FastAPI Server Design

The FastAPI app loads the model once on startup along with the scaler to avoid repeated disk I/O. On each POST /predict request: (1) Validate input using Pydantic schemas (amount>0, merchant_id>0, customer_id>0, etc.). (2) Preprocess features (apply scaler transformation). (3) Run XGBoost inference. (4) Return prediction (0 or 1) + fraud_probability (0-1) + latency_ms + model_version. Metrics emitted: predictions_total counter (tracks total predictions by model_version and result), prediction_latency_seconds histogram (latency distribution), model_version_info gauge (tracks versions), data_drift_score gauge (0-1 drift metric).

Health endpoint GET /health checks: model is loaded in memory, database connection works, returns HTTP 200 if healthy. Metrics endpoint GET /metrics exposes Prometheus-format metrics. OpenAPI endpoint GET /docs provides auto-generated interactive documentation.

Request schema: PredictionRequest with fields amount (float, >0), merchant_id (int, >0), customer_id (int, >0), transaction_count (int, ≥0), avg_amount (float, ≥0). Response schema: PredictionResponse with is_fraud (0 or 1), fraud_probability (0-1), latency_ms (float), model_version (string like "v1").

FastAPI was chosen for high performance with async support, built-in Pydantic validation, auto OpenAPI docs, easy Prometheus metrics integration, and strong type hints. Latency breakdown: request validation 0.2ms + feature preprocessing 0.5ms + XGBoost inference 15-30ms + metrics emission 0.3ms + response serialization 0.5ms = <50ms p95. Optimization: model loads once, predictions cached when possible, features pre-scaled.

---

## Kubernetes Deployment

### Deployment Strategy

The Kubernetes Deployment uses: metadata name="fraud-detection-api" in namespace="mlops", spec.replicas=3 (minimum), strategy=RollingUpdate with maxSurge=1 (max 1 extra pod during update) and maxUnavailable=0 (zero-downtime deployments). Container uses image ml-api:v1, exposes port 8000.

Liveness probe at GET /health with initialDelaySeconds=10 and periodSeconds=10 detects dead pods - if fails 3 times, pod is restarted. Readiness probe at GET /health with initialDelaySeconds=5 and periodSeconds=5 determines if pod can receive traffic. Resource requests: 512Mi memory + 250m CPU (guaranteed). Resource limits: 1Gi memory + 500m CPU (hard caps).

### Horizontal Pod Autoscaler

HPA is configured: minReplicas=3, maxReplicas=10, scales on CPU utilization (target 70%) and memory utilization (target 80%). When load increases, HPA gradually adds pods. When load decreases, it removes pods after cooldown period. Example scaling: 100 predictions/sec = 3 pods @ 35% CPU (healthy), 500 predictions/sec = 3 pods @ 100% CPU → triggers scale-up to 5 pods @ 60% CPU (stable), 1000 predictions/sec = 5 pods @ 100% CPU → triggers scale-up to 9 pods @ 55% CPU (stable), 1200+ predictions/sec = 10 pods (max) @ 80%+ CPU (need to upgrade pod spec or partition traffic).

---

## Monitoring & Observability

### Prometheus Metrics

The API exposes metrics at /metrics endpoint. Metrics include: predictions_total (Counter tracking total predictions with labels model_version and result), prediction_latency_seconds (Histogram with buckets 0.01, 0.025, 0.05, 0.1, 0.5, 1.0 seconds), model_accuracy (Gauge showing current AUC by model_version), data_drift_score (Gauge 0-1 per feature).

Prometheus queries: rate(predictions_total[1m]) for request rate in requests/sec, histogram_quantile(0.95, prediction_latency_seconds_bucket) for p95 latency, rate(predictions_total{result="error"}[5m]) for error rate, sum(rate(container_cpu_usage_seconds_total[1m])) by (pod) for pod CPU usage.

### Grafana Dashboard

The dashboard displays: (1) Request Rate panel showing rate(predictions_total[1m]), (2) Latency panels for p50/p95/p99 using histogram_quantile, (3) Model Accuracy gauge showing model_accuracy, (4) Data Drift panel showing data_drift_score, (5) Pod Count showing count(kube_pod_info), (6) Pod CPU showing sum by (pod) of CPU usage, (7) Pod Memory showing sum by (pod) of memory usage, (8) Error Rate showing rate(predictions_total{result="error"}[5m]).

---

## Auto-retraining Pipeline

### Airflow DAG Design

The daily DAG runs at 2 AM and executes: (1) Check Data Drift - query last 30 days predictions, calculate current AUC, compare vs baseline, if accuracy degraded >5% proceed else skip. (2) Retrain Model - load new features from PostgreSQL, train XGBoost, track in MLflow, version with DVC. (3) Evaluate New Model - compute metrics on test set, compare vs baseline AUC, if better promote else reject. (4) Deploy New Model - update Kubernetes config, trigger K8s rollout, monitor health. (5) Validate Deployment - run health checks, smoke tests, verify metrics are normal. (6) Alert if Issues - send Slack notification, PagerDuty alert if critical.

Drift detection logic: get last 30 days of predictions from database, calculate current_auc = roc_auc_score(actual, predicted_prob), get baseline_auc from previous tracking, calculate degradation_pct = (baseline_auc - current_auc) / baseline_auc * 100, if degradation_pct > 5 trigger retraining else skip.

---

## Design Decisions

### 1. Streaming vs Batch Feature Engineering

Decision: Streaming (Spark Streaming). Reasoning: sub-minute feature freshness, real-time predictions needed, scales to thousands of transactions/sec. Alternative (Batch with Spark SQL): stale features (hours old), prediction lag, but simpler to operate.

### 2. XGBoost vs Neural Network

Decision: XGBoost. Reasoning: fast inference <50ms, interpretable with feature importance, proven for tabular data, no GPU needed. Alternative (Neural Network): slower inference 100-500ms, black box, requires GPU, better for text/images.

### 3. Single Model vs Multi-Model Ensemble

Decision: Single XGBoost (for now). Reasoning: simpler deployment, faster inference, easier monitoring. Future: could ensemble XGBoost + SHAP + calibrated probabilities.

### 4. PostgreSQL vs Data Warehouse

Decision: PostgreSQL. Reasoning: features queryable in SQL, fast for <100GB, ACID guarantees, pgVector extension for future embeddings. Alternative (Snowflake/BigQuery): better for 1TB+ data but higher cost and network latency.

### 5. Docker + Kubernetes vs Serverless

Decision: Docker + Kubernetes. Reasoning: full control, cost-effective at scale, stateless containers, multi-cloud portable. Alternative (AWS Lambda/Google Cloud Functions): no ops needed but cold start latency 1-5s, vendor lock-in, expensive for high volume.

---

## Scalability Considerations

### Kafka Scaling

Current: 3 brokers, 12 partitions. Scaling path: 1K TPS needs 3 brokers/6 partitions, 10K TPS needs 6 brokers/12 partitions, 100K TPS needs 12 brokers/24 partitions, 1M TPS needs 24+ brokers/48+ partitions. Strategy: increase brokers (add machines), increase partitions (better parallelism), adjust replication factor (3 for production).

### Spark Scaling

Current: 4 executors, 2GB memory each. 10K TPS: 4 executors, 2GB memory, 2 cores each, <1min process latency. 100K TPS: 16 executors, 4GB memory, 4 cores each, <1min process latency. Kubernetes auto-scaling: use Spark Operator or YARN with --executor-instances 16 --executor-cores 4 --executor-memory 4g.

### API Scaling

Current: 3 replicas, <50ms latency, 10K predictions/sec. Throughput calculation: per pod 1000 reqs/sec (2 cores, 512MB) = 3 pods = 3000 reqs/sec, max 10 pods = 10,000 reqs/sec. For 100K reqs/sec: need 10+ K8s nodes, consider sharding by customer_id, or multi-region deployment.

### PostgreSQL Scaling

Current: single instance, <10GB. Scaling path: <100GB use single instance + read replicas, >100GB use partitioning by customer_id, >1TB use sharding or data warehouse (Snowflake). Partitioning example: CREATE TABLE features_shard_1 PARTITION OF features FOR VALUES FROM (1) TO (100000).

---

## Failure Modes & Mitigation

### Kafka Broker Failure

If one broker fails: replication factor 3 means only 1 broker can fail, auto-recovery adds new broker to cluster. Monitoring: kafka_server_broker_up{job="kafka"}.

### Spark Job Failure

If streaming job crashes: Kubernetes restarts pod, checkpoint recovery resumes from last committed state, dead letter queue handles unparseable messages. Monitoring: kubectl get pods -n mlops | grep spark, kubectl logs spark-streaming-job-xyz.

### Model Degradation

If model accuracy drops: Airflow detects drift daily, auto-retrains if degradation >5%, A/B tests new model before full rollout, can quickly rollback to previous version. Monitoring: model_accuracy{model_version="v1"}.

### API Pod Crash

If pod becomes unhealthy: Kubernetes liveness probe detects, auto-restarts pod, HPA brings up new pod if needed, requests routed to healthy pods. Monitoring: kubectl get pods -n mlops, kubectl describe pod <pod-name>.

### Database Connection Pool Exhaustion

If all DB connections in use: connection pool limit 20, new requests fail if exceeded, monitor active connections, scale up replicas if sustained high load. Monitoring: SELECT count(*) FROM pg_stat_activity;.

---

## Performance Tuning

### Inference Latency Optimization

Baseline 50ms. Optimizations: model quantization 50ms → 30ms, batch processing 30ms → 15ms (if tolerable), GPU inference 15ms → 5ms (if volume justifies), ONNX export 5ms → 2ms (lower level).

### Feature Computation Optimization

Current: 1-minute window every 1 second. Optimizations: reduce window 1 min → 30 sec (lower latency), increase batching 1 sec → 5 sec (lower CPU), materialize common aggregates (speed up), use approximate algorithms (approx-count, HyperLogLog).

---

## Cost Optimization

### Cloud Infrastructure Costs

Component costs: Kafka $500/mo (3 brokers), Spark $800/mo (16 executors), PostgreSQL $300/mo (managed), Kubernetes $400/mo (3 nodes), Monitoring $200/mo (self-hosted), Total $2,200/mo. Optimizations: use managed Kafka (AWS MSK, Confluent Cloud), on-demand + spot instances for Spark, reserved instances for PostgreSQL, auto-scaling + spot for Kubernetes, managed monitoring (DataDog, New Relic). Target: optimize to $1,000/mo.

---

## Security Considerations

### Network Security

API → LoadBalancer (TLS 1.3) → Kubernetes Service (internal DNS) → Pod (no encryption, same network) → PostgreSQL (encrypted connection + auth).

### Secrets Management

Kubernetes Secrets: database credentials, API keys, model registry tokens. Alternative: HashiCorp Vault for more advanced needs.

### Model Registry Security

MLflow: restrict access (basic auth or OAuth), version all changes, audit logs for model transitions.

---

**Last Updated**: December 2025  
**Version**: 1.0  
**Status**: Production-Ready
