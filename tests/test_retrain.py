from pathlib import Path
import pytest
import mlflow
from src.retrain import is_retraining_triggered, execute_retraining
from src.train import run_experiment_and_register_best


def test_retraining_execution_with_flag(tmp_path: Path):
    flag_path = tmp_path / "drift_detected.flag"
    flag_path.touch()

    # Use isolated local tracking URI for test
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_test.db"
    res = execute_retraining(
        tracking_uri=tracking_uri,
        experiment_name="test-churn-retrain",
        model_name="test-retrain-model",
        flag_path=flag_path,
        cleanup_flag=True
    )
    assert res is not None
    assert res["status"] == "success"
    assert "run_id" in res
    assert not flag_path.exists()  # Flag was cleaned up


def test_train_experiment_and_register(tmp_path: Path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_train_test.db"
    res = run_experiment_and_register_best(
        tracking_uri=tracking_uri,
        experiment_name="test-churn-experiment",
        model_name="test-prod-model"
    )
    assert res["model_name"] == "test-prod-model"
    assert res["stage"] == "Production"
    assert res["best_run_id"] is not None
