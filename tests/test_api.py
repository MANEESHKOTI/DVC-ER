from fastapi.testclient import TestClient
import pytest
from src.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """
    Requirement 8:
    GET /health returns 200 OK and {"status": "ok"}
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid():
    """
    Requirement 9:
    POST /predict with valid payload returns 200 OK and {"prediction": <int>}
    """
    payload = {
        "feature1": 0.5,
        "feature2": 1.2,
        "category_feature": "A"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in (0, 1)


def test_predict_endpoint_invalid_type():
    """
    Requirement 10:
    POST /predict with malformed payload returns 422 Unprocessable Entity
    """
    payload = {
        "feature1": "this-should-be-a-float"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_fields():
    """
    Requirement 10:
    POST /predict with missing fields returns 422 Unprocessable Entity
    """
    payload = {
        "feature1": 1.5
        # feature2 and category_feature are missing
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_empty_payload():
    """
    Requirement 10:
    POST /predict with empty payload returns 422 Unprocessable Entity
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422
