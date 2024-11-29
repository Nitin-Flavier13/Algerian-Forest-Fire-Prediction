import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, render_template, request

application = Flask(__name__)
app = application

# import our model and scaler
elasticNetCV = pickle.load(open('models/elasticNetCV.pkl','rb'))
standardScaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/",methods=['GET','POST'])
def predict():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        input_data = standardScaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        pred_fwi = elasticNetCV.predict(input_data)

        return render_template('home.html',results=pred_fwi[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)