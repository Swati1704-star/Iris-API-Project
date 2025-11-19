from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/iris_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is working!"})

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
    app.run(port=5000)
