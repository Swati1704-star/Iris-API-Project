import requests
import pickle
import numpy as np

# Load reference model (local original model)
with open("model/iris_model.pkl", "rb") as f:
    reference_model = pickle.load(f)

# Sample test input
features = [5.1, 3.5, 1.4, 0.2]
test_array = np.array(features).reshape(1, -1)

# Prediction from reference model
reference_prediction = reference_model.predict(test_array)[0]

# Prediction from deployed model (API)
url = "http://localhost:5000/predict"
payload = {"features": features}
response = requests.post(url, json=payload)

if response.status_code == 200:
    deployed_prediction = response.json()["prediction"][0]

    print("Reference Model Prediction :", reference_prediction)
    print("Deployed Model Prediction  :", deployed_prediction)

    if reference_prediction == deployed_prediction:
        print("\n✅ MATCH — Deployment is correct")
    else:
        print("\n❌ MISMATCH — Deployment has an issue")
else:
    print("API Error:", response.text)

