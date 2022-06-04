
import sys
import os
sys.path.append(os.getcwd())

from flask import Flask, jsonify, request
from flask_cors.decorator import cross_origin

from prediction_service import PredictionService

app = Flask(__name__) 
prediction_service = PredictionService()

@app.route('/predict')
@cross_origin()
def predict():
    text = request.args["text"]
    result = prediction_service.predict(text)
    return jsonify(message=result)
