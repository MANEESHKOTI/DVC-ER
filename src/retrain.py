import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

try:
    from src.config import settings
    from src.train import create_model_pipeline, load_train_test_data, evaluate_model
except ImportError:
    from config import settings
    from train import create_model_pipeline, load_train_test_data, evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_retraining_triggered(flag_path: Path = settings.DRIFT_FLAG_PATH) -> bool:
    """Check if the trigger condition (drift_detected.flag) is met."""
    return flag_path.exists()


def execute_retraining(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None,
    flag_path: Path = settings.DRIFT_FLAG_PATH,
    force: bool = False,
    cleanup_flag: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Executes conditional model retraining based on drift detection flag.
    - If drift_detected.flag is missing and not force: skips retraining.
    - If drift_detected.flag is present: trains a new model candidate,
      logs the run, registers a new model version in MLflow registry.
    """
    if not force and not is_retraining_triggered(flag_path):
        logger.warning(
            f"Retraining aborted: Trigger flag '{flag_path.name}' does not exist. "
            "No data drift has been flagged."
        )
        return None

    logger.info(f"Trigger condition met ('{flag_path.name}' found). Commencing retraining pipeline...")

    tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    experiment_name = experiment_name or settings.EXPERIMENT_NAME
    model_name = model_name or settings.MODEL_NAME

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load fresh/updated dataset
    X_train, y_train, X_test, y_test = load_train_test_data()

    # Train updated model with refined hyperparameters
    params = {
        "n_estimators": 160,
        "learning_rate": 0.08,
        "max_depth": 5,
        "retrained": True
    }

    pipeline = create_model_pipeline(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"]
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Log retraining run in MLflow
    with mlflow.start_run(run_name="retrained_candidate") as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_test.head(2),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        logger.info(
            f"Retraining Run completed (ID: {run_id}) - F1 Score: {metrics['f1_score']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

    # Register new model version in MLflow Model Registry (Requirement 12 & 13)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_uri = f"runs:/{run_id}/model"

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    new_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    logger.info(
        f"Registered new model version '{new_version.version}' for '{model_name}' in MLflow Registry."
    )

    if cleanup_flag and flag_path.exists():
        flag_path.unlink()
        logger.info(f"Cleaned up flag '{flag_path.name}'.")

    return {
        "status": "success",
        "run_id": run_id,
        "model_version": new_version.version,
        "metrics": metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Trigger model retraining pipeline conditionally.")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of drift flag.")
    parser.add_argument("--cleanup", action="store_true", help="Delete drift flag after retraining.")
    args = parser.parse_args()

    result = execute_retraining(force=args.force, cleanup_flag=args.cleanup)
    if result is None:
        sys.exit(0)
    else:
        logger.info(f"Retraining finished successfully: {result}")
        sys.exit(0)


if __name__ == "__main__":
    main()
