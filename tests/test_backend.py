import pytest
from fastapi.testclient import TestClient
from src.inference.api_server import app

client = TestClient(app)

FEATURE_COLS = [
    "heart_rate",
    "hr_rolling_mean",
    "hr_rolling_std",
    "acc_magnitude",
    "act_rest",
    "act_walk",
    "act_brisk",
    "act_run",
]

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["features"] == FEATURE_COLS

def test_predict_ok():
    payload = {
        "heart_rate": 90.0,
        "hr_rolling_mean": 89.2,
        "hr_rolling_std": 1.5,
        "acc_magnitude": 9.7,
        "act_rest": 1,
        "act_walk": 0,
        "act_brisk": 0,
        "act_run": 0
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["pred_class"] in (0, 1, 2)

def test_predict_invalid_onehot():
    payload = {
        "heart_rate": 90.0,
        "hr_rolling_mean": 89.2,
        "hr_rolling_std": 1.5,
        "acc_magnitude": 9.7,
        "act_rest": 1,
        "act_walk": 1,  # invalid: 2 activities
        "act_brisk": 0,
        "act_run": 0
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any("Exactly one of act_rest" in err.get("msg", "")
               for err in detail
               )

def test_predict_extra_field():
    payload = {
        "heart_rate": 90.0,
        "hr_rolling_mean": 89.2,
        "hr_rolling_std": 1.5,
        "acc_magnitude": 9.7,
        "act_rest": 1,
        "act_walk": 0,
        "act_brisk": 0,
        "act_run": 0,
        "hour_sin": 0.0  # extra field
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422

def test_predict_from_sensor_ok():
    payload = {
        "heart_rate": 82.0,
        "hr_window": [78.0, 80.0, 81.0, 82.0, 83.0],
        "acc_x": 0.2,
        "acc_y": 9.7,
        "acc_z": 0.3,
        "activity_group": "rest",
        "hour_of_day": 9.0
    }
    resp = client.post("/predict-from-sensor", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["pred_class"] in (0, 1, 2)

def test_predict_from_sensor_invalid_group():
    payload = {
        "heart_rate": 82.0,
        "hr_window": [78.0, 80.0, 81.0, 82.0, 83.0],
        "acc_x": 0.2,
        "acc_y": 9.7,
        "acc_z": 0.3,
        "activity_group": "invalid_group",
        "hour_of_day": 9.0
    }
    resp = client.post("/predict-from-sensor", json=payload)
    assert resp.status_code == 422

def test_predict_from_sensor_activity_id():
    payload = {
        "heart_rate": 112.0,
        "hr_window": [108.0, 110.0, 111.0, 112.0, 113.0],
        "acc_x": 0.4,
        "acc_y": 9.7,
        "acc_z": 0.9,
        "activity_id": 4,
        "hour_of_day": 10.0
    }
    resp = client.post("/predict-from-sensor", json=payload)
    assert resp.status_code == 200

def test_predict_from_sensor_device_code():
    payload = {
        "heart_rate": 138.0,
        "hr_window": [130.0, 133.0, 135.0, 137.0, 138.0],
        "acc_x": 0.9,
        "acc_y": 9.8,
        "acc_z": 1.8,
        "device_activity_code": 2,
        "hour_of_day": 15.0
    }
    resp = client.post("/predict-from-sensor", json=payload)
    assert resp.status_code == 200