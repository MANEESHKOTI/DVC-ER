from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management for the MLOps pipeline.
    All parameters are overridable via environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database Configuration
    POSTGRES_USER: str = "mlflow_user"
    POSTGRES_PASSWORD: str = "mlflow_password"
    POSTGRES_DB: str = "mlflow_db"

    # MLflow Tracking & Registry
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    EXPERIMENT_NAME: str = "churn-prediction"
    MODEL_NAME: str = "production-model"
    MODEL_STAGE: str = "Production"

    # API Configuration
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "raw" / "customer_churn.csv"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    TRAIN_DATA_PATH: Path = BASE_DIR / "data" / "processed" / "train.csv"
    TEST_DATA_PATH: Path = BASE_DIR / "data" / "processed" / "test.csv"
    REFERENCE_DATA_PATH: Path = BASE_DIR / "data" / "processed" / "reference.csv"
    PRODUCTION_SIM_DATA_PATH: Path = BASE_DIR / "data" / "processed" / "production_stream.csv"

    # Reports & Validations
    REPORTS_DIR: Path = BASE_DIR / "reports"
    DRIFT_REPORT_PATH: Path = BASE_DIR / "reports" / "data_drift_report.html"
    VALIDATION_RESULTS_PATH: Path = (
        BASE_DIR / "great_expectations" / "uncommitted" / "validations" / "churn_suite" / "results.json"
    )

    # Retraining Flag
    DRIFT_FLAG_PATH: Path = BASE_DIR / "drift_detected.flag"


# Global settings singleton
settings = Settings()
