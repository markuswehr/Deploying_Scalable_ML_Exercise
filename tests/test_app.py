from fastapi.testclient import TestClient

import json

from main import app

client = TestClient(app)


def test_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome_message": "Welcome to my app."}


def test_model_inference():
    r = client.post("/model_inference")
    assert r.status_code == 200


def test_get_malformed():
    r = client.post("/model")
    assert r.status_code != 200


def test_model_inference(test_data):
    #client = TestClient(app)
    response = client.post("/model_inference", json=test_data)
    assert response.status_code == 200
    assert response.json()["Prediction"] in ["<=50K", ">50k"]
