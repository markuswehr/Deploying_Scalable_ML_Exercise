# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pickle
import os
import pandas as pd
from ml.data import process_data
from ml.model import train_model, get_sliced_preformance, inference, compute_model_metrics

# Add code to load in the data.
root_path = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(root_path, "../data/census.csv"), skipinitialspace = True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Save encoder and binarizer
pickle.dump(encoder, open(os.path.join(root_path, "encoder.sav"), "wb"))
pickle.dump(encoder, open(os.path.join(root_path, "lb.sav"), "wb"))

# Train and save a model.
model = train_model(X_train, y_train)
# save the model to disk
pickle.dump(model, open(os.path.join(root_path, "final_model.sav"), "wb"))

# Output 
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
y_pred = inference(model=model, X=X_test)

# Calculate overall model performance
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print("-----------Overall model performance-----------")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Beta: {fbeta}")

# calculated sliced model performance
get_sliced_preformance(test, label="salary", y_pred=y_pred, slice_cols=cat_features)
