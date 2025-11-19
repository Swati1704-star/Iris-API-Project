from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running!"}), 200

# Load model
MODEL_PATH = os.path.join("model", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

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

        prediction = model.predict(features)[0]
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()
