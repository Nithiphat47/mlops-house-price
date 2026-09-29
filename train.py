import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# 1. เตรียมข้อมูล
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("house_price_prediction")

def train_and_log(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        
        signature = infer_signature(X_train, preds)
        # ใช้ cloudpickle เพื่อเลี่ยงปัญหา Untrusted types
        mlflow.sklearn.log_model(model, "model", signature=signature, serialization_format="cloudpickle")
        return mlflow.active_run().info.run_id, rmse

# 2. เทรนโมเดล 2 แบบ
print("Training models...")
lr_id, lr_rmse = train_and_log(LinearRegression(), "LinearRegression")
rf_id, rf_rmse = train_and_log(RandomForestRegressor(n_estimators=50, random_state=42), "RandomForest")

# 3. เลือกโมเดลที่ดีที่สุดลงทะเบียน
best_id = lr_id if lr_rmse < rf_rmse else rf_id
mlflow.register_model(f"runs:/{best_id}/model", "house-price-model")
print(f"Registered Best Model (Run ID: {best_id})")