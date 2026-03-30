"""
MediScan AI — Backend tests
Run: pytest tests/ -v
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app
import pytest

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "models_loaded" in data

def test_cors_headers(client):
    r = client.get("/health")
    assert "Access-Control-Allow-Origin" in r.headers

def test_predict_diabetes_missing_fields(client):
    r = client.post("/predict/diabetes",
                    data=json.dumps({"Glucose": 120}),
                    content_type="application/json")
    # Either 404 (model not loaded) or 400 (missing fields)
    assert r.status_code in (400, 404)

def test_predict_unknown_disease(client):
    r = client.post("/predict/unknowndisease",
                    data=json.dumps({}),
                    content_type="application/json")
    assert r.status_code == 404

def test_predict_diabetes_full(client):
    """Only runs if model is loaded."""
    payload = {
        "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 168, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    r = client.post("/predict/diabetes",
                    data=json.dumps(payload),
                    content_type="application/json")
    if r.status_code == 404:
        pytest.skip("Model not loaded — run training script first")
    assert r.status_code == 200
    data = r.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "metrics" in data
    assert "prescription" in data
    assert data["prediction"] in ("Positive", "Negative")
    assert 0 <= data["confidence"] <= 100
    assert "accuracy" in data["metrics"]

def test_predict_image_no_data(client):
    r = client.post("/predict/pneumonia",
                    data=json.dumps({}),
                    content_type="application/json")
    assert r.status_code in (400, 404)