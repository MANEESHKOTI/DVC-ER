import logging
import urllib.request
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

try:
    from src.config import settings
except ImportError:
    from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global model cache
_loaded_model = None


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictionInput(BaseModel):
    feature1: float = Field(..., description="Numerical feature 1 (float)")
    feature2: float = Field(..., description="Numerical feature 2 (float)")
    category_feature: str = Field(..., description="Categorical feature (string, e.g., 'A', 'B', 'C')")

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "feature1": 0.5,
                "feature2": 1.2,
                "category_feature": "A"
            }
        }
    }


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Model prediction (0 or 1)")


def is_mlflow_reachable(uri: str) -> bool:
    """Fast check whether the MLflow server is responding to avoid long client timeouts."""
    try:
        req = urllib.request.Request(uri, headers={"User-Agent": "FastAPI-MLOps"})
        with urllib.request.urlopen(req, timeout=0.8) as response:
            return response.status in (200, 301, 302)
    except Exception:
        return False


def get_production_model(force_reload: bool = False):
    """
    Load production-stage model from MLflow Model Registry.
    Falls back gracefully and instantly to a trained pipeline if registry is offline or initializing.
    """
    global _loaded_model
    if _loaded_model is not None and not force_reload:
        return _loaded_model

    tracking_uri = settings.MLFLOW_TRACKING_URI
    model_name = settings.MODEL_NAME
    model_stage = settings.MODEL_STAGE

    if is_mlflow_reachable(tracking_uri):
        try:
            logger.info(f"Loading '{model_name}:{model_stage}' from MLflow registry at {tracking_uri}...")
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"models:/{model_name}/{model_stage}"
            _loaded_model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Successfully loaded '{model_name}:{model_stage}' from MLflow Model Registry.")
            return _loaded_model
        except Exception as e:
            logger.warning(f"Could not load '{model_name}:{model_stage}' from MLflow registry: {e}.")
    else:
        logger.info(f"MLflow server at {tracking_uri} is not reachable. Using local trained fallback model.")

    # Fallback to local training/pipeline
    try:
        from src.train import create_model_pipeline, load_train_test_data
        X_train, y_train, _, _ = load_train_test_data()
        fallback_pipeline = create_model_pipeline()
        fallback_pipeline.fit(X_train, y_train)
        _loaded_model = fallback_pipeline
        logger.info("Initialized local fallback model pipeline successfully.")
        return _loaded_model
    except Exception as fallback_err:
        logger.error(f"Failed to initialize fallback model: {fallback_err}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    logger.info("Initializing FastAPI application and preloading model...")
    get_production_model()
    yield
    logger.info("Shutting down FastAPI application.")


app = FastAPI(
    title="MLOps Production Prediction Service",
    description="Serves predictions from the production-stage MLflow registered model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """
    Health check endpoint for Docker and orchestration monitoring.
    Returns: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict(payload: PredictionInput) -> Dict[str, int]:
    """
    Prediction endpoint accepting feature payloads and returning predictions.
    Input validation handled by Pydantic; invalid requests automatically return 422.
    """
    model = get_production_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production model is currently unavailable or initializing."
        )

    # Convert request payload to DataFrame matching training format
    input_df = pd.DataFrame([{
        "feature1": payload.feature1,
        "feature2": payload.feature2,
        "category_feature": str(payload.category_feature).strip().upper()
    }])

    try:
        raw_pred = model.predict(input_df)[0]
        prediction_val = int(raw_pred)
        return {"prediction": prediction_val}
    except Exception as err:
        logger.error(f"Prediction inference error: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(err)}"
        )
