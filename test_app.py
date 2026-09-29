from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict():
    sample_data = {
        "MedInc": 8.32, "HouseAge": 41.0, "AveRooms": 6.98,
        "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.55,
        "Latitude": 37.88, "Longitude": -122.23
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "predicted_price_100k" in response.json()
