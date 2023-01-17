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
    below_50, above_50 = test_data
    r = client.post("/model_inference", json=below_50.to_dict())

    assert r.status_code == 200
    assert r.json() == {"Prediction": ["<=50K"]}

def test_model_inference_above_50(test_data):
    below_50, above_50 = test_data
    r = client.post("/model_inference", json=above_50.to_dict())

    assert r.status_code == 200
    assert r.json() == {"Prediction": [">50K"]}