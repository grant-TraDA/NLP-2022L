
import sys
import os
sys.path.append(os.getcwd())

from flask import Flask, jsonify
from flask_cors.decorator import cross_origin

from prediction_service import PredictionService

app = Flask(__name__) 
prediction_service = PredictionService()

@app.route('/predict/<text>')
@cross_origin()
def predict(text):
    result = prediction_service.predict(text)
    return jsonify(message=result)
