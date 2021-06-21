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
from flassgger import swagger

app = Flask(__name__)
swagger(app)

@app.route('/')
def welcome():
    return "Welcome all"

pickle_in = open('classifier.pkl','rb')
classifiers = pickle.load(pickle_in)

@app.route('/predict')
def predict_note_authenticate():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifiers.predict([[variance,skewness,curtosis,entropy]])
    return 'The value is' + str(prediction)
 
@app.route('/file_prediction', methods=['POST'])
def predict_note_file():
    df_test = pd.read_csv(request.files.get('tstfile'))
    prediction = classifiers.predict(df_test)
    return "The predicted value is" + str(list(prediction))

if __name__ == '__main__':
    app.run()

