import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

try:
    from src.config import settings
except ImportError:
    from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_model_pipeline(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    random_state: int = 42
) -> Pipeline:
    """
    Build scikit-learn Pipeline with integrated ColumnTransformer.
    This guarantees raw payloads with feature1, feature2, category_feature
    can be passed directly to the pipeline.
    """
    numeric_features = ["feature1", "feature2"]
    categorical_features = ["category_feature"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )
    return pipeline


def load_train_test_data(
    train_path: Path = settings.TRAIN_DATA_PATH,
    test_path: Path = settings.TEST_DATA_PATH
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and test datasets from disk."""
    if not train_path.exists() or not test_path.exists():
        from src.data_processing import process_and_split
        logger.info("Processed data not found; triggering data processing pipeline...")
        process_and_split()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = ["feature1", "feature2", "category_feature"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    return X_train, y_train, X_test, y_test


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate pipeline and compute accuracy, precision, recall, and f1_score."""
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    return metrics


def train_and_log_run(
    experiment_name: str,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: Optional[str] = None
) -> Tuple[str, float, Pipeline]:
    """Execute a single training run, log parameters/metrics/model to MLflow."""
    pipeline = create_model_pipeline(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"]
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        # Log parameters (learning_rate required by spec)
        mlflow.log_params(params)
        # Log metrics (f1_score required by spec)
        mlflow.log_metrics(metrics)

        # Log model artifact with sample input
        input_example = X_test.head(2)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        logger.info(
            f"Run {run_name} (ID: {run_id}) - F1 Score: {metrics['f1_score']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

    return run_id, metrics["f1_score"], pipeline


def run_experiment_and_register_best(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes an MLflow experiment with at least 3 distinct runs,
    selects the champion model, registers it in the Model Registry,
    and transitions its stage to 'Production'.
    """
    tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    experiment_name = experiment_name or settings.EXPERIMENT_NAME
    model_name = model_name or settings.MODEL_NAME

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"Connected to MLflow at {tracking_uri}, Experiment: {experiment_name}")

    X_train, y_train, X_test, y_test = load_train_test_data()

    # Define at least 3 distinct configurations as required by Requirement 6
    configs = [
        {"n_estimators": 50, "learning_rate": 0.02, "max_depth": 3, "name": "baseline_run"},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4, "name": "standard_run"},
        {"n_estimators": 150, "learning_rate": 0.10, "max_depth": 5, "name": "tuned_run"},
    ]

    best_run_id = None
    best_f1 = -1.0
    best_pipeline = None

    for idx, cfg in enumerate(configs, 1):
        params = {
            "n_estimators": cfg["n_estimators"],
            "learning_rate": cfg["learning_rate"],
            "max_depth": cfg["max_depth"]
        }
        run_id, f1, pipeline = train_and_log_run(
            experiment_name=experiment_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            run_name=cfg["name"]
        )
        if f1 > best_f1 or best_run_id is None:
            best_f1 = f1
            best_run_id = run_id
            best_pipeline = pipeline

    logger.info(f"Best run selected: {best_run_id} with F1-score: {best_f1:.4f}")

    # Register best model in Model Registry and transition to 'Production' stage
    client = MlflowClient(tracking_uri=tracking_uri)
    model_uri = f"runs:/{best_run_id}/model"

    try:
        reg_model = client.create_registered_model(model_name)
        logger.info(f"Created registered model '{model_name}'")
    except Exception:
        logger.info(f"Registered model '{model_name}' already exists.")

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run_id
    )
    logger.info(f"Created model version {model_version.version} for '{model_name}'")

    # Transition to Production stage (Requirement 7)
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=settings.MODEL_STAGE,  # 'Production'
        archive_existing_versions=True
    )
    logger.info(f"Transitioned model version {model_version.version} to '{settings.MODEL_STAGE}' stage.")

    return {
        "best_run_id": best_run_id,
        "best_f1": best_f1,
        "model_version": model_version.version,
        "model_name": model_name,
        "stage": settings.MODEL_STAGE
    }


if __name__ == "__main__":
    run_experiment_and_register_best()
