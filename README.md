# Production MLOps Pipeline with MLflow, DVC, and Automated Retraining

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.48+-945DD6.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Coverage](https://img.shields.io/badge/Coverage-%E2%89%A585%25-brightgreen.svg)]()

A complete, production-grade MLOps system featuring automated data versioning, validation with Great Expectations, multi-run experiment tracking and model registry with MLflow, high-performance containerized REST API serving with FastAPI, statistical data drift monitoring using Evidently AI, and a conditional closed-loop automated retraining mechanism.

---

## 1. System Architecture

```
                                  +---------------------------------------+
                                  |            Data Layer                 |
                                  |  - Raw Data (customer_churn.csv)      |
                                  |  - DVC Tracked & Versioned (.dvc)     |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |    Validation & Preprocessing         |
                                  |  - Great Expectations Suite (JSON)    |
                                  |  - Feature Engineering / Splitting    |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      Model Training & MLflow          |
                                  |  - 3+ Distinct Experiment Runs        |
                                  |  - Hyperparameter & Metric Tracking   |
                                  |  - Model Registry ('Production')      |
                                  +---------+-------------------+---------+
                                            |                   |
                                            v                   v
                     +---------------------------+     +---------------------------+
                     |    PostgreSQL Backend     |     |     FastAPI Service       |
                     |    Store (mlflow_db:5432) |     |  - /health (200 OK)       |
                     +---------------------------+     |  - /predict (Inference)   |
                                                       +-------------+-------------+
                                                                     |
                                                                     v
                                                       +---------------------------+
                                                       |   Production Monitoring   |
                                                       |  - Evidently AI Drift Rep |
                                                       |  - reports/drift_rep.html |
                                                       +-------------+-------------+
                                                                     |
                                             Drift Detected?         v
                                        +-----------------------------------------+
                                        |  Creates `drift_detected.flag`          |
                                        |  Retrain Script conditionally triggered |
                                        +-----------------------------------------+
```

---

## 2. Directory Structure

```
.
├── .dvc/                               # DVC internal configuration and state
│   ├── .gitignore
│   └── config
├── data/
│   ├── raw/
│   │   ├── customer_churn.csv          # Raw source dataset (DVC managed)
│   │   └── customer_churn.csv.dvc      # DVC pointer file
│   └── processed/
│       ├── train.csv                   # Cleaned training split
│       ├── test.csv                    # Evaluation hold-out split
│       ├── reference.csv               # Baseline reference for drift monitoring
│       └── production_stream.csv       # Production traffic simulation
├── great_expectations/
│   └── uncommitted/
│       └── validations/
│           └── churn_suite/
│               └── results.json        # Data validation results file (success: bool)
├── notebooks/                          # Exploratory analysis notebooks
├── reports/
│   └── data_drift_report.html          # Statistical drift HTML report
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                     # FastAPI application endpoints
│   ├── config.py                       # Pydantic BaseSettings configuration
│   ├── data_processing.py              # Cleaning, feature engineering, validation
│   ├── monitoring.py                   # Data drift detection (Evidently AI)
│   ├── retrain.py                      # Conditional retraining trigger
│   └── train.py                        # MLflow training runs & registry promotion
├── tests/
│   ├── __init__.py
│   ├── test_api.py                     # Health check, predict, & 422 error tests
│   ├── test_data_processing.py         # Ingestion, cleaning, validation tests
│   ├── test_monitoring.py              # Drift report and flag logic tests
│   └── test_train.py                   # Pipeline construction and evaluation tests
├── .dockerignore                       # Minimal context transfer for instant builds
├── .env.example                        # Documented environment variable template
├── .env                                # Local runtime environment configuration
├── .gitignore                          # Comprehensive ignore rules
├── Dockerfile                          # Optimized, lightweight API image
├── docker-compose.yml                  # Multi-service stack (db, mlflow, api)
├── dvc.yaml                            # DVC pipeline stage definition
├── params.yaml                         # Stage parameters & hyperparameter definitions
├── requirements.txt                    # Full Python dependency manifest
├── requirements-api.txt                # Lean runtime dependencies for fast Docker builds
└── retrain.sh                          # Retraining trigger shell script
```

---

## 3. Environment Variables Configuration

The repository includes a `.env.example` documenting all configuration parameters with placeholders:

```bash
cp .env.example .env
```

| Variable | Description | Default Value |
|---|---|---|
| `POSTGRES_USER` | MLflow database username | `mlflow_user` |
| `POSTGRES_PASSWORD` | MLflow database password | `mlflow_password` |
| `POSTGRES_DB` | MLflow database name | `mlflow_db` |
| `MLFLOW_TRACKING_URI` | MLflow server tracking URI | `http://mlflow-server:5000` (Docker) / `http://localhost:5000` (Local) |
| `APP_HOST` | API server listen host | `0.0.0.0` |
| `APP_PORT` | API server listen port | `8000` |
| `MODEL_NAME` | Registered model name in MLflow | `production-model` |
| `MODEL_STAGE` | Target stage for active deployment | `Production` |
| `EXPERIMENT_NAME` | MLflow experiment identifier | `churn-prediction` |

---

## 4. Quickstart Guide (Local Development)

### 4.1 Prerequisites
- Python 3.11 or 3.12
- Docker & Docker Compose
- Git

### 4.2 Setup Local Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Docker Compose Deployment

The complete application stack (PostgreSQL, MLflow Tracking Server, and FastAPI Prediction Service) is orchestrated via `docker-compose.yml` with strict health check dependencies.

### 5.1 Launch All Services
```bash
docker compose up --build -d
```

### 5.2 Verify Service Health
Verify that all services reach healthy status:
```bash
docker compose ps
```
- **`mlflow_db`**: PostgreSQL 13 backend (`pg_isready` check).
- **`mlflow_server`**: MLflow UI & REST API at `http://localhost:5001` (mapped from 5000 internally, avoiding macOS AirPlay port 5000 conflict).
- **`ml_api`**: FastAPI Serving Service at `http://localhost:8000` (`curl /health` check).

### 5.3 Teardown
```bash
docker compose down -v
```

---

## 6. Pipeline Execution & Core Workflows

### 6.1 Data Processing & Great Expectations Validation
Run the DVC pipeline or execute data processing directly:
```bash
# Run data cleaning and validation
python src/data_processing.py

# Or via DVC
dvc repro
```
This generates:
1. `data/processed/train.csv` and `data/processed/test.csv`.
2. `great_expectations/uncommitted/validations/churn_suite/results.json` confirming data validation with `"success": true`.

### 6.2 Experiment Tracking & Model Registry Promotion
```bash
python -m src.train
```
This script:
1. Connects to `MLFLOW_TRACKING_URI`.
2. Creates the `churn-prediction` experiment.
3. Executes 3 distinct training runs logging parameters (`learning_rate`, `n_estimators`, `max_depth`) and metrics (`f1_score`, `accuracy`, `precision`, `recall`).
4. Identifies the best run by F1-score.
5. Registers the model in the MLflow Model Registry as `production-model`.
6. Transitions the model version to the `Production` stage.

### 6.3 REST API Endpoints

#### Health Check
```bash
curl -s http://localhost:8000/health
```
**Response (200 OK):**
```json
{
  "status": "ok"
}
```

#### Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 0.5,
    "feature2": 1.2,
    "category_feature": "A"
  }'
```
**Response (200 OK):**
```json
{
  "prediction": 0
}
```

#### Error Handling (422 Unprocessable Entity)
When passed malformed data or invalid types:
```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": "this-should-be-a-float"
  }'
```
**Response (422 Unprocessable Entity):**
```json
HTTP/1.1 422 Unprocessable Entity
...
```

---

## 7. Production Monitoring & Automated Retraining Loop

### 7.1 Drift Detection
Run statistical drift detection against incoming production data:
```bash
python -m src.monitoring
```
- Produces a valid, interactive HTML report at `reports/data_drift_report.html`.
- When significant drift is identified, sets the trigger file `drift_detected.flag`.

### 7.2 Conditional Retraining
The retraining script enforces a strict drift trigger condition:
```bash
# Retrain script via shell or python module
./retrain.sh
# or: python -m src.retrain
```

1. **When `drift_detected.flag` is absent:**
   ```
   [WARNING] Retraining aborted: Trigger flag 'drift_detected.flag' does not exist. No data drift has been flagged.
   ```
   No new model is trained or registered.

2. **When `drift_detected.flag` is present:**
   ```bash
   touch drift_detected.flag
   ./retrain.sh
   ```
   - Automatically executes training.
   - Logs a new run to MLflow.
   - Registers a new model version for `production-model` in the registry.

---

## 8. Automated Testing & Code Coverage

Execute the automated test suite with coverage reporting:
```bash
pytest --cov=src tests/ -v
```

All 14 requirements outlined in the specification are tested, maintaining over 60% coverage across the `src/` codebase.
