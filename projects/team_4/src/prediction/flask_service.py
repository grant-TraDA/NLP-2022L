
import sys
import os
sys.path.append(os.getcwd())

from flask import Flask
from prediction_service import PredictionService

app = Flask(__name__)
prediction_service = PredictionService()

@app.route('/predict/<text>')
def predict(text):
    return prediction_service.predict(text)
