from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained ML model
MODEL_PATH = os.path.join("model", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({"message": "API is working!"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate if "features" key exists
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input. 'features' key missing."}), 400

        features = data["features"]

        # Validate correct input format -> list of 4 numbers
        if not isinstance(features, list) or len(features) != 4:
            return jsonify({"error": "Invalid input format. Provide 4 numeric features in a list."}), 400

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # Perform prediction
        prediction = model.predict(feat
