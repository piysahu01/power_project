from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load('C:/Users/Piyush Sahu/power_model.pkl')

class PowerInput(BaseModel):
    temperature: float
    humidity: float
    hour_of_day: int
    previous_day_usage: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "🚀 Power Usage Prediction API is running!"}

@app.post("/predict")
def predict_power(data: PowerInput):
    input_data = np.array([[data.temperature, data.humidity, data.hour_of_day, data.previous_day_usage]])
    prediction = model.predict(input_data)
    return {"predicted_power_usage": prediction[0]}
