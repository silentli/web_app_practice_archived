#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:02:57 2023
"""

#%%
import pandas as pd
from flask import Flask, request, render_template
from pycaret.regression import load_model, predict_model
import folium
from datetime import datetime


app = Flask(__name__)
model = load_model('./model/park_pycaret_2012_pipeline')
data = pd.read_csv('./data/street_sensor_longlat.csv')


@app.route('/', methods= ['POST', 'GET'])
def index():
    return 'Go to /predict'


@app.route('/predict', methods= ['POST', 'GET'])
def predict():
    map = folium.Map(location=[37.79016837, -122.415677],tiles='cartodbpositron',
                     zoom_start=14,width='75%', height='75%')
    
    if request.method == 'POST':
        # date = request.form.get('Date')
        # hour = request.form.get('Hour')
        # holiday = request.form.get('Holiday')
        # rain = request.form.get('Rain')
        #
        # if not all([date, hour, holiday, rain]):
        #     return "Please fill in all fields correctly!", 400

        pre=pd.DataFrame(data['BLOCK_ID'])
        date = request.form.get('Date',type=str)
        day = datetime.strptime(date,'%d/%m/%Y').weekday()
        pre['Dayofweek'] = day
        pre['Hour'] = request.form.get('Hour',type=int)
        pre['holiday'] = request.form.get('Holiday',type=int)
        if day<=4:
            pre['DAY_TYPE'] = 0
        else:
            pre['DAY_TYPE'] = 1
        pre['Precipitation']= request.form.get('Rain',type=int)

        
        prediction = predict_model(model,data=pre)
        prediction= prediction.merge(data,on='BLOCK_ID')

        
        for i in range(0,len(prediction)):
            if prediction['prediction_label'].iloc[i]>0.85:
                folium.Marker([prediction['lat'].iloc[i],prediction['lng'].iloc[i]],
                           tooltip=prediction['STREET_BLOCK'].iloc[i],
                           popup=prediction['prediction_label'].iloc[i].round(2),
                           icon=folium.Icon( icon='fa-car', prefix='fa',color='red')
                           ).add_to(map)
            if (prediction['prediction_label'].iloc[i]>0.6) and (prediction['prediction_label'].iloc[i]<=0.85):
                folium.Marker([prediction['lat'].iloc[i],prediction['lng'].iloc[i]],
                           tooltip=prediction['STREET_BLOCK'].iloc[i],
                           popup=prediction['prediction_label'].iloc[i].round(2),
                           icon=folium.Icon( icon='fa-car', prefix='fa',color='orange')
                           ).add_to(map)        
            if prediction['prediction_label'].iloc[i]<=0.6:
                folium.Marker([prediction['lat'].iloc[i],prediction['lng'].iloc[i]],
                           tooltip=prediction['STREET_BLOCK'].iloc[i],
                           popup=prediction['prediction_label'].iloc[i].round(2),
                           icon=folium.Icon( icon='fa-car', prefix='fa',color='green')
                           ).add_to(map)            

        map.save('templates/map.html')
                
        return render_template('index.html')
    if request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
