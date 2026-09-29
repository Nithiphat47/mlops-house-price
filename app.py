from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import mlflow
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

# ค้นหาโมเดลที่ดีที่สุดอัตโนมัติจาก MLflow โดยไม่ต้องล็อกเวอร์ชัน
experiment = mlflow.get_experiment_by_name("house_price_prediction")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
best_run_id = runs.loc[runs['metrics.rmse'].idxmin()]['run_id']
model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict")
def predict(features: HouseFeatures):
    # ใช้ model_dump() แทน dict() เพื่อแก้ Warning ของ Pydantic V2
    data = pd.DataFrame([features.model_dump()])
    pred = model.predict(data)
    return {"predicted_price_100k": round(float(pred[0]), 2)}