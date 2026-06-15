from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fruit MLOps API")

model = joblib.load("model.pkl")

class FruitRequest(BaseModel):
    feature_x: float

@app.get("/")
def home():
    return {"message": "Fruit API running"}

@app.post("/predict")
def predict(data: FruitRequest):
    prediction = model.predict(np.array([[data.feature_x]]))
    return {"prediction": float(prediction[0])}
