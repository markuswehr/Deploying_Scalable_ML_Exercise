import pytest
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session")
def data():
    df = pd.DataFrame(np.random.randint(0, 400, size=(200, 4), ), columns=["A", "B", "C", "D"])
    df["y_true"] = np.random.randint(0, 2, size=(200,))
    y = df.pop("y_true")
    X = df

    return X, y


@pytest.fixture(scope="session")
def model(data):
    X, y = data
    model = RandomForestClassifier(random_state=23)
    model.fit(X, y)

    return model

@pytest.fixture(scope='session')
def test_data(data):
    X, y = data

    below_50 = X[X["salary"] == "<=50K"].iloc[0].drop("salary")
    above_50 = X[X["salary"] == ">50K"].iloc[0].drop("salary")

    return below_50, above_50
