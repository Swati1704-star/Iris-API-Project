from flask import Flask, request
import json
from datetime import datetime

app = Flask(__name__)

def log(message):
    with open("api_logs.txt", "a") as file:
        file.write(message + "\n\n")

@app.before_request
def log_request():
    data = f"""
---- Incoming Request ({datetime.now()}) ----
URL: {request.url}
Method: {request.method}
Headers: {dict(request.headers)}
Body: {request.get_data(as_text=True)}
---------------------------------------------
"""
    log(data)

@app.after_request
def log_response(response):
    data = f"""
---- Outgoing Response ({datetime.now()}) ----
Status Code: {response.status_code}
Response: {response.get_data(as_text=True)}
---------------------------------------------
"""
    log(data)
    return response
