import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train import (
    create_model_pipeline,
    evaluate_model,
    train_and_log_run
)


def test_create_model_pipeline():
    pipeline = create_model_pipeline(n_estimators=10, max_depth=2)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_evaluate_model():
    pipeline = create_model_pipeline(n_estimators=10, max_depth=2)
    X = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature2": [2.0, 4.0, 1.0, 3.0, 5.0, 2.0],
        "category_feature": ["A", "B", "A", "B", "A", "B"]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])

    pipeline.fit(X, y)
    metrics = evaluate_model(pipeline, X, y)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
