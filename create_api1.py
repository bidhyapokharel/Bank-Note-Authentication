# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

@app.route('/')
def welcome():
    return "Welcome all"

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/predict', methods=["GET"])
def predict_note_authenticate():
    '''Let's Authenticate the Bank Notes
    This is using docstrings for specifications.
    ---
    parameters:
      - name:variance
        in:query
        type:number
        required:true
      - name:skewness
        in:query
        type:number
        required:true
      - name:curtosis
        in:query
        type:number
        required:true
      - name:entropy
        in:query
        type:number
        required:true
    responses:
        200:
            description: The Output values
    '''
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return 'The value is' + str(prediction)
 
@app.route('/file_prediction', methods=["POST"])
def predict_note_file():
    '''Let's Authenticate the Bank Notes
    This is using docstrings for specifications. 
    ---
    parameters:
        - name:tstfile
          in:formData
          type:file
          required:true
    responses:
        200:
            description: The output value
    '''
    df_test = pd.read_csv(request.files.get('tstfile'))
    prediction = classifier.predict(df_test)
    return "The predicted value is" + str(list(prediction))

if __name__ == '__main__':
    app.run()

