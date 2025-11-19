from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join("model", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    # MUST return JSON so response.json works in the test
    return jsonify({"message": "API is working!"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # The test sends these 4 keys directly
        features = np.array([
            data["Sepal Length"],
            data["Sepal Width"],
            data["Petal Length"],
            data["Petal Width"],
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        return jsonify({"prediction": str(prediction)}), 200

    except Exception as e:
        # If anything is wrong with input, return 400
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
