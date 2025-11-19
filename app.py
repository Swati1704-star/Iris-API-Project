from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Correct model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Iris API is running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([
            data["Sepal Length"],
            data["Sepal Width"],
            data["Petal Length"],
            data["Petal Width"]
        ]).reshape(1, -1)

        pr
