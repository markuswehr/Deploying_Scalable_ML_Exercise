from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome_message": "Welcome to my app."}


def test_model_inference(test_data_below_50):
    r = client.post("/model_inference", json=test_data_below_50)
    assert r.status_code == 200


def test_get_malformed():
    r = client.post("/model")
    assert r.status_code != 200


def test_model_inference_below_50(test_data_below_50):
    #client = TestClient(app)
    response = client.post("/model_inference", json=test_data_below_50)
    assert response.status_code == 200
    assert response.json()["Prediction"] == "<=50K"


def test_model_inference_above_50(test_data_above_50):
    #client = TestClient(app)
    response = client.post("/model_inference", json=test_data_above_50)
    assert response.status_code == 200
    assert response.json()["Prediction"] == ">50K"
