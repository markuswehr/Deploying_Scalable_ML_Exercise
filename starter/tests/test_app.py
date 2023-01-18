from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome_message": "Welcome to my app."}


def test_model_inference():
    r = client.get("/model_inference")
    assert r.status_code == 200


def test_get_malformed():
    r = client.get("/model")
    assert r.status_code != 200


def test_model_inference_below_50(test_data):
    X, y = test_data
    r = client.post("/model_inference", json=X.to_dict())

    assert r.status_code == 200
    assert "<=50K" in r.json().get("Prediction")


def test_model_inference_above_50(test_data):
    X, y = test_data
    r = client.post("/model_inference", json=X.to_dict())

    assert r.status_code == 200
    assert ">50K" in r.json().get("Prediction")
