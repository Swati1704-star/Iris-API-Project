import json
from app import app

client = app.test_client()

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json["message"] == "API is working!"

def test_predict_valid():
    sample = {
        "Sepal Length": 5.1,
        "Sepal Width": 3.5,
        "Petal Length": 1.4,
        "Petal Width": 0.2
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert "prediction" in response.json

def test_predict_invalid():
    sample = {"Sepal Length": 5.1}
    response = client.post("/predict", json=sample)
    assert response.status_code == 400
