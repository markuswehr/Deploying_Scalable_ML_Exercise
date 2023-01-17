from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, inference, compute_model_metrics

import pandas as pd


def test_train_model(data):
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference(model, data):
    X, y = data
    y_pred = inference(model, X)

    assert len(y_pred) == len(y)
    assert y_pred.any() == 1


def test_compute_model_metrics(model, data):
    X, y = data
    y_pred = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, y_pred)

    assert precision != 0.0
    assert recall != 0.0
    assert fbeta != 0.0
