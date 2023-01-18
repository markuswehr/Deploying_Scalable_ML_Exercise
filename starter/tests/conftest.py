import pytest
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data


root_path = os.path.dirname(os.path.abspath(__file__))


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

@pytest.fixture(scope='session')
def test_data(data):
    # Add code to load in the data.
    data = pd.read_csv(os.path.join(root_path, "../data/census_cleaned.csv"))
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Proces the test data with the process_data function.
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    return X, y
