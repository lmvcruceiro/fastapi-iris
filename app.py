# app.py
from typing import Optional
from fastapi import FastAPI
import joblib
from pydantic import BaseModel, confloat

# some documentation in markdown
description = """
## Documentation
**ℹ️ Read carefully before using**

This api allows you to predict the type of Iris plant given a list of features.
The features should be:
* sepal length in cm
* sepal width in cm
* petal length in cm
* petal width in cm

"""

# create FastAPI app and load model
app = FastAPI(
    title="IRIS Classification",
    description=description,
    version="0.1"
)

# create FastAPI app and load model
model = joblib.load("model.joblib")


# We'll take this in:
class Features(BaseModel):
    sepal_length: confloat(ge=0.0, le=1.0) # ensures values  are between 0 and 1
    sepal_width: confloat(ge=0.0, le=1.0)
    petal_length: confloat(ge=0.0, le=1.0)
    petal_width: confloat(ge=0.0, le=1.0)

        # with an example
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 0.2,
                "sepal_width": 0.5,
                "petal_length": 0.8,
                "petal_width": 1.0,
            }
        }

# We'll respond something like this:
class Response(BaseModel):
    setosa_probability: confloat(ge=0.0, le=1.0)
    versicolor_probability: confloat(ge=0.0, le=1.0)
    virginica_probability: confloat(ge=0.0, le=1.0)

        # with an example
    class Config:
        json_schema_extra = {
            "example": {
                "setosa_probability": 0.7,
                "versicolor_probability": 0.1,
                "virginica_probability": 0.2,
            }
        }

# create an endpoint that receives POST requests
# and returns predictions
# the endpoint
@app.post("/predict/", response_model=Response)
def predict(features: Features):
    feature_list = [
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.sepal_width,
    ]
    predictions = model.predict_proba([feature_list])[-1]
    predictions_clean = Response(
        setosa_probability=predictions[0],
        versicolor_probability=predictions[1],
        virginica_probability=predictions[2],
    )
    return predictions_clean