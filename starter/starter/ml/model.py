from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : str
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def get_sliced_preformance(data, label, y_pred, slice_cols):
    data["y_pred"] = y_pred
    label_category_0 = data[label].unique()[0]
    data["y_true"] = [0 if x==label_category_0 else 1 for x in data[label]]
    for feature in slice_cols:
        print(f"\n-------------{feature.upper()}-------------\n")
        for category in data[feature].unique():
            y_true = data[data[feature]==category]["y_true"]
            y_pred = data[data[feature]==category]["y_pred"]
            precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
            print("-------------------------------------------")
            print(f"Category: {category}")
            print("-------------------------------------------")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F-Beta: {fbeta}")
