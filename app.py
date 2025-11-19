from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "API is running", 200   # Matching unit test expectation

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract features exactly as the test sends them
    try:
        features = np.array([
            data["Sepal Length"],
            data["Sepal Width"],
            data["Petal Length"],
            data["Petal Width"]
        ]).reshape(1, -1)
    except Exception:
        return jsonify({"error": "Invalid input"}), 400

    try:
        prediction = model.predict(features)[0]
        return jsonify({"prediction": str(prediction)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
