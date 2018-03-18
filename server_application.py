# -*- coding: utf-8 -*-
""" Flask webservice application accepting HTTP POST request with JSON data
corresponding to a list of Sentence to be classified.
See README for a detailed specification of
the JSON data excepted for the request.

Example:
    The webservice can for example be deployed using gunicorn:
    $gunicorn server:app
"""
from flask import Flask, request, jsonify
from sentiment_classifier import SENTIMENTClassifier, SENTIMENTClassifierInputError

#Create an instance of a SENTIMENTClassifier that initizalizes and controls the
#state of the tensorflow graph
REQUEST_HANDLER = SENTIMENTClassifier()

#Create the Flask application handling HTTP requests
app = Flask(__name__)

@app.route('/sentiment/classify', methods=['POST'])
def classify_sentiment():
    """Unpacks the JSON data passed with the POST request and forwards it to the
   SENTIMENTClassifier for classification"""
    if request.method == 'POST':
        resp = jsonify([])
        try:
            classifications = REQUEST_HANDLER.classify(request.json['requests'])
            data = {
                'responses'  : classifications,
            }
            resp = jsonify(data)
            resp.status_code = 200
            return resp
        except SENTIMENTClassifierInputError as excep:
            resp = bad_input("Invalid input detected: {}"
                             .format(excep))
            return resp
        #Handle Internal Server Errors
        except Exception as excep:
            resp = bad_input("Unexpected server API error: {}"
                             .format(excep))
            return resp

def bad_input(message):
    """Returns a 404 status code JSON response with the provided message"""
    response = jsonify({'message': message})
    response.status_code = 404
    return response
