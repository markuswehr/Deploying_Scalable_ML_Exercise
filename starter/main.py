# Put the code for your API here.
from typing import Union, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import pickle
import pandas as pd

from starter.ml.model import inference
from starter.ml.data import process_data

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

model = pickle.load(open("starter/final_model.sav", "rb"))
encoder = pickle.load(open("starter/encoder.sav", "rb"))

app = FastAPI()

class CensusItem(BaseModel):
    age: Optional[Union[int, list]] = [30, 30, 30, 30]
    workclass: Optional[Union[str, list]] = ["Private", "Private", "Private", "Private"]
    fnlgt: Optional[Union[int, list]] = [215646, 215646, 215646, 215646]
    education: Optional[Union[str, list]] = ["Masters", "Masters", "Masters", "Masters"]
    education_num: Optional[Union[int, list]] = Field([13, 13, 13, 13], alias="education-num")
    marital_status: Optional[Union[str, list]] = Field(["Never-married", "Never-married", "Never-married", "Never-married"], alias="marital-status")
    occupation: Optional[Union[str, list]] = ["Tech-support", "Tech-support", "Tech-support", "Tech-support"]
    relationship: Optional[Union[str, list]] = ["Not-in-family", "Not-in-family", "Not-in-family", "Not-in-family"]
    race: Optional[Union[str, list]] = ["White", "White", "White", "White"]
    sex: Optional[Union[str, list]] = ["Male", "Male", "Male", "Male"]
    capital_gain: Optional[Union[int, list]] = Field([2174, 2174, 2174, 2174], alias="capital-gain")
    capital_loss: Optional[Union[int, list]] = Field([0, 0, 0, 0], alias="capital-loss")
    hours_per_week: Optional[Union[int, list]] = Field([40, 40, 40, 40], alias="hours-per-week")
    native_country: Optional[Union[str, list]] = Field(["Germany", "Canada", "Portugal", "United-States"], alias="native-country")

@app.get("/")
async def welcome():
    return {"welcome_message": "Welcome to my app."}

@app.post("/model_inference")
async def model_inference(data: CensusItem):
    data_df = pd.DataFrame.from_dict(data.dict(by_alias=True))
    X, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,
    )
    pred = inference(model=model, X=X)
    if pred[0] == 0:
        pred_label = "<=50K"
    else:
        pred_label = ">50K"
    return {"Prediction": pred_label}
