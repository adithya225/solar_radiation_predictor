import numpy as np
from flask import Flask, request, jsonify, render_template
from pandas import DataFrame
import pickle

app = Flask(__name__)
model = pickle.load(open('xgb_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    print(features)
    year = int(features[0])
    month = int(features[1])
    day = int(features[2])
    hour = int(features[3])
    minute = int(features[4])
    cloudtype = int(features[5])
    dewpoint = float(features[6])
    solarzenithangle = float(features[7])
    surfacealdebo = float(features[8])
    windspeed = float(features[9])
    precwater = float(features[10])
    winddirection = int(features[11])
    relativehumidity = float(features[12])
    temperature = float(features[13])
    pressure = int(features[14])
    final_features = []
    a = [final_features]
    #a=[]
    #a = 
    #print(a)
    final_features.append(year)
    final_features.append(month)
    final_features.append(day)
    final_features.append(hour)
    final_features.append(minute)
    final_features.append(cloudtype)
    final_features.append(dewpoint)
    final_features.append(solarzenithangle)
    final_features.append(surfacealdebo)
    final_features.append(windspeed)
    final_features.append(precwater)
    final_features.append(winddirection)
    final_features.append(relativehumidity)
    final_features.append(temperature)
    final_features.append(pressure)
    print(len(final_features))
    df = DataFrame(a, columns=None)
    prediction = model.predict(df)
    output = round(prediction[0], 2)    
    return render_template('index.html',prediction_text = output)
 


if __name__ == "__main__":
    app.run(debug=True)
