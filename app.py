# app.py
from typing import Optional
from fastapi import FastAPI
from fastapi import Response
import joblib
from pydantic import BaseModel, confloat
import pickle
from typing import Any

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
    sepal_length: confloat(ge=0.0, le=10) # ensures values  are between 0 and 1
    sepal_width: confloat(ge=0.0, le=10)
    petal_length: confloat(ge=0.0, le=10)
    petal_width: confloat(ge=0.0, le=10)
    A1662707202: confloat(ge=0.0)
    A1662708602: confloat(ge=0.0)
    A1662708702: confloat(ge=0.0)
    A2132704801: confloat(ge=0.0)
    A2560107300: confloat(ge=0.0)
    A2572700000: confloat(ge=0.0)
    A2760101414: confloat(ge=0.0)
    A2760101614: confloat(ge=0.0)
    A2760106714: confloat(ge=0.0)
    A6540107406: confloat(ge=0.0)
    Arrival_Ship_in_Bahnterminal_Savannah: confloat(ge=0.0, le=1)
    Arrival_Ship_in_Hafen_Charleston: confloat(ge=0.0, le=1)
    Arrival_Truck_in_Hafen_Aguascalientes: confloat(ge=0.0, le=1)
    Arrival_Truck_in_Hafen_Antwerpen: confloat(ge=0.0, le=1)
    Arrival_Truck_in_Hafen_Bremerhaven: confloat(ge=0.0, le=1)
    Arrival_Truck_in_Hafen_Charleston: confloat(ge=0.0, le=1)
    Arrival_Truck_in_Hafen_Hamburg: confloat(ge=0.0, le=1)
    Arrival_Truck_in_Hafen_Rotterdam: confloat(ge=0.0, le=1)
    Container_Closed_in_Unknown: confloat(ge=0.0, le=1)
    Departure_Ship_in_Hafen_Antwerpen: confloat(ge=0.0, le=1)
    Departure_Ship_in_Hafen_Bremerhaven: confloat(ge=0.0, le=1)
    Departure_Ship_in_Hafen_Hamburg: confloat(ge=0.0, le=1)
    Departure_Ship_in_Hafen_Rotterdam: confloat(ge=0.0, le=1)
    Departure_Truck_in_Bahnterminal_Savannah: confloat(ge=0.0, le=1)
    Departure_Truck_in_Hafen_Altamira: confloat(ge=0.0, le=1)
    Departure_Truck_in_Hafen_Charleston: confloat(ge=0.0, le=1)
    Goods_Receipt_Dock_in_Werk_Tuscaloosa: confloat(ge=0.0, le=1)
    Loading_Ship_in_Hafen_Antwerpen: confloat(ge=0.0, le=1)
    Loading_Ship_in_Hafen_Bremerhaven: confloat(ge=0.0, le=1)
    Loading_Ship_in_Hafen_Hamburg: confloat(ge=0.0, le=1)
    Loading_Ship_in_Hafen_Rotterdam: confloat(ge=0.0, le=1)
    Loading_Truck_in_Consolidation_Center_Speyer: confloat(ge=0.0, le=1)
    New_Scheduling_in_Bahnterminal_Savannah: confloat(ge=0.0, le=1)
    New_Scheduling_in_Hafen_Charleston: confloat(ge=0.0, le=1)
    Registration_Yard_in_Werk_Tuscaloosa: confloat(ge=0.0, le=1)
    Unloading_Ship_in_Bahnterminal_Savannah: confloat(ge=0.0, le=1)
    Unloading_Ship_in_Hafen_Charleston: confloat(ge=0.0, le=1)

        # with an example
    class Config:
        json_schema_extra = {
            "example": {
        "A1662707202": 0,
        "A1662708602": 0,
        "A1662708702": 0,
        "A2132704801": 102,
        "A2560107300": 0,
        "A2572700000": 6,
        "A2760101414": 0,
        "A2760101614": 0,
        "A2760106714": 0,
        "A6540107406": 0,
        "Arrival_Ship_in_Bahnterminal_Savannah": 0,
        "Arrival_Ship_in_Hafen_Charleston": 0,
        "Arrival_Truck_in_Hafen_Aguascalientes": 0,
        "Arrival_Truck_in_Hafen_Antwerpen": 0,
        "Arrival_Truck_in_Hafen_Bremerhaven": 0,
        "Arrival_Truck_in_Hafen_Charleston": 0,
        "Arrival_Truck_in_Hafen_Hamburg": 0,
        "Arrival_Truck_in_Hafen_Rotterdam": 0,
        "Container_Closed_in_Unknown": 0,
        "Departure_Ship_in_Hafen_Antwerpen": 0,
        "Departure_Ship_in_Hafen_Bremerhaven": 1,
        "Departure_Ship_in_Hafen_Hamburg": 0,
        "Departure_Ship_in_Hafen_Rotterdam": 0,
        "Departure_Truck_in_Bahnterminal_Savannah": 0,
        "Departure_Truck_in_Hafen_Altamira": 0,
        "Departure_Truck_in_Hafen_Charleston": 1,
        "Goods_Receipt_Dock_in_Werk_Tuscaloosa": 1,
        "Loading_Ship_in_Hafen_Antwerpen": 0,
        "Loading_Ship_in_Hafen_Bremerhaven": 1,
        "Loading_Ship_in_Hafen_Hamburg": 0,
        "Loading_Ship_in_Hafen_Rotterdam": 0,
        "Loading_Truck_in_Consolidation_Center_Speyer": 0,
        "New_Scheduling_in_Bahnterminal_Savannah": 0,
        "New_Scheduling_in_Hafen_Charleston": 0,
        "Registration_Yard_in_Werk_Tuscaloosa": 1,
        "Unloading_Ship_in_Bahnterminal_Savannah": 0,
        "Unloading_Ship_in_Hafen_Charleston": 1
            }
        }


@app.post("/predict/")
def predict(features: Features):
    feature_list = [
        features.A1662707202,
        features.A1662708602,
        features.A1662708702,
        features.A2132704801,
        features.A2560107300,
        features.A2572700000,
        features.A2760101414,
        features.A2760101614,
        features.A2760106714,
        features.A6540107406,
        features.Arrival_Ship_in_Bahnterminal_Savannah,
        features.Arrival_Ship_in_Hafen_Charleston,
        features.Arrival_Truck_in_Hafen_Aguascalientes,
        features.Arrival_Truck_in_Hafen_Antwerpen,
        features.Arrival_Truck_in_Hafen_Bremerhaven,
        features.Arrival_Truck_in_Hafen_Charleston,
        features.Arrival_Truck_in_Hafen_Hamburg,
        features.Arrival_Truck_in_Hafen_Rotterdam,
        features.Container_Closed_in_Unknown,
        features.Departure_Ship_in_Hafen_Antwerpen,
        features.Departure_Ship_in_Hafen_Bremerhaven,
        features.Departure_Ship_in_Hafen_Hamburg,
        features.Departure_Ship_in_Hafen_Rotterdam,
        features.Departure_Truck_in_Bahnterminal_Savannah,
        features.Departure_Truck_in_Hafen_Altamira,
        features.Departure_Truck_in_Hafen_Charleston,
        features.Goods_Receipt_Dock_in_Werk_Tuscaloosa,
        features.Loading_Ship_in_Hafen_Antwerpen,
        features.Loading_Ship_in_Hafen_Bremerhaven,
        features.Loading_Ship_in_Hafen_Hamburg,
        features.Loading_Ship_in_Hafen_Rotterdam,
        features.Loading_Truck_in_Consolidation_Center_Speyer,
        features.New_Scheduling_in_Bahnterminal_Savannah,
        features.New_Scheduling_in_Hafen_Charleston,
        features.Registration_Yard_in_Werk_Tuscaloosa,
        features.Unloading_Ship_in_Bahnterminal_Savannah,
        features.Unloading_Ship_in_Hafen_Charleston
    ]
    prediction = model.predict([feature_list])

    return prediction