# Put the code for your API here.
from typing import Union, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import pickle
import os
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

root_path = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(root_path, "model/final_model.sav"), "rb"))
encoder = pickle.load(open(os.path.join(root_path, "model/encoder.sav"), "rb"))

app = FastAPI()

class CensusItem(BaseModel):
    age: int = Field(default=30)
    workclass: str = Field(default="Private")
    fnlgt: int = Field(default=215646)
    education: str = Field(default="Masters")
    education_num: int = Field(default=18, alias="education-num")
    marital_status: str = Field(default="Never-married", alias="marital-status")
    occupation: str = Field(default="Tech-support")
    relationship: str = Field(default="Not-in-family")
    race: str = Field(default="White")
    sex: str = Field(default="Male")
    capital_gain: int = Field(default=2174, alias="capital-gain")
    capital_loss: int = Field(default=0, alias="capital-loss")
    hours_per_week: int = Field(default=40, alias="hours-per-week")
    native_country: str = Field(default="Germany", alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "workclass": "Private",
                "fnlgt": 215646,
                "education": "Masters",
                "education_num": "education-num",
                "marital_status": "marital-status",
                "occupation": "Tech-support",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "Germany",
            }
        }


@app.get("/")
async def welcome():
    return {"welcome_message": "Welcome to my app."}

@app.post("/model_inference")
async def model_inference(data: CensusItem):
    data_df = pd.DataFrame.from_dict([data.dict(by_alias=True)])
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
