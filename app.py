from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({"message": "API is working!"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", None)

        if features is None:
            return jsonify({"error": "No features provided"}), 400

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]

        return jsonify({"prediction": str(prediction)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
