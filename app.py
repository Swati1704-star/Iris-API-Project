from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")  # Change path if model is inside folder

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([[data["Sepal Length"], data["Sepal Width"], data["Petal Length"], data["Petal Width"]]])
        prediction = model.predict(features)
        return jsonify({"prediction": str(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 👇 IMPORTANT FIX FOR CI (pytest)
# Do NOT run Flask server when pytest imports this file
if __name__ == "__main__":
    app.run(debug=False)
