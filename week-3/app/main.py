import pandas as pd
from fastapi import FastAPI
import joblib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR/"models"/"best_model.joblib"



from app.schemas import PassengerInput

app = FastAPI()

model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "ML Prediction API"}


@app.post("/predict")
def predict(data: PassengerInput):
    input_data = pd.DataFrame([{
        "Pclass": data.Pclass,
        "Sex": data.Sex,
        "Age": data.Age,
        "SibSp": data.SibSp,
        "Parch": data.Parch,
        "Fare": data.Fare,
        "Embarked": data.Embarked,
    }])

    prediction = model.predict(input_data)


    return {
        "prediction": int(prediction[0])
    }