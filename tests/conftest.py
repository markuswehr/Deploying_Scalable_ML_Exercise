import pytest
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session")
def data():
    df = pd.DataFrame(np.random.randint(0, 400, size=(200, 4), ), columns=["A", "B", "C", "D"])
    df["y_true"] = np.random.randint(2, size=(200,))
    y = df.pop("y_true")
    X = df

    return X, y


@pytest.fixture(scope="session")
def model(data):
    X, y = data
    model = RandomForestClassifier(random_state=23)
    model.fit(X, y)

    return model


@pytest.fixture(scope="session")
def test_data_above_50():
    input_data = {
        "age": 60,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "Master",
        "education-num": 18,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    return input_data


@pytest.fixture(scope="session")
def test_data_below_50():
    input_data = {
        "age": 18,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Cuba"
    }

    return input_data