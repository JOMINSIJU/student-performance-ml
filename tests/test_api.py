from fastapi.testclient import TestClient

import app as service

client = TestClient(service.app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction():
    service.load_artifacts()

    if service.model is None:
        return  # Allows tests to run without training artifacts.

    response = client.post("/predict", json={
        "study_hours": 5,
        "attendance": 85,
        "previous_score": 70
    })

    assert response.status_code == 200
    assert 0 <= response.json()["predicted_score"] <= 100

def test_invalid_input():
    response = client.post("/predict", json={
        "study_hours": -5,
        "attendance": 85,
        "previous_score": 70
    })
    assert response.status_code == 422

def test_missing_input():
    response = client.post("/predict", json={
        "study_hours": 5,
        "attendance": 85
    })
    assert response.status_code == 422

def test_readiness_without_model():
    old_model = service.model
    old_metadata = service.metadata

    try:
        service.model = None
        service.metadata = None

        response = client.get("/ready")
        assert response.status_code == 503
    finally:
        service.model = old_model
        service.metadata = old_metadata