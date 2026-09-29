from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="House Price API")

# กำหนดรูปแบบข้อมูล Input
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# โหลดโมเดลเวอร์ชันล่าสุดที่เราพึ่งเทรนเสร็จ
model = mlflow.pyfunc.load_model("models:/house-price-model/1")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict")
def predict(features: HouseFeatures):
    data = pd.DataFrame([features.dict()])
    pred = model.predict(data)
    return {"predicted_price_100k": round(float(pred[0]), 2)}