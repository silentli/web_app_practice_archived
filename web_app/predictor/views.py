from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
import folium
from .utils.loaders import load_model_file, load_data_file
from pycaret.regression import predict_model
import pandas as pd

# Load the model and data using the loader module
model = load_model_file()
data = load_data_file()

def predict(request):
    sf_map = folium.Map(location=[37.79016837, -122.415677],
                     tiles='cartodbpositron',
                     zoom_start=14, width='75%', height='75%')

    sf_map.save('predictor/templates/sf_map.html')

    if request.method == 'POST':
        # Retrieve form data
        date = request.POST.get('Date')
        hour = request.POST.get('Hour')
        holiday = request.POST.get('Holiday')
        rain = request.POST.get('Rain')

        # Check for missing fields
        if not all([date, hour, holiday, rain]):
            return HttpResponse("Please fill in all fields correctly!", status=400)

        # Prepare data for prediction
        pre = pd.DataFrame(data['BLOCK_ID'])
        day = datetime.strptime(date, '%d/%m/%Y').weekday()
        pre['Dayofweek'] = day
        pre['Hour'] = int(hour)
        pre['holiday'] = int(holiday)
        pre['Precipitation'] = int(rain)

        if day <= 4:
            pre['DAY_TYPE'] = 0
        else:
            pre['DAY_TYPE'] = 1

        # Make predictions
        prediction = predict_model(model, data=pre)
        prediction = prediction.merge(data, on='BLOCK_ID')

        # Add markers to the sf_map
        for i in range(len(prediction)):
            prediction_label = prediction['prediction_label'].iloc[i]
            color = ('red' if prediction_label > 0.85 else
                     'orange' if prediction_label > 0.6 else 'green')

            folium.Marker([
                prediction['lat'].iloc[i],
                prediction['lng'].iloc[i]
            ],
                tooltip=prediction['STREET_BLOCK'].iloc[i],
                popup=round(prediction_label, 2),
                icon=folium.Icon(icon='fa-car', prefix='fa', color=color)
            ).add_to(sf_map)

        # Save the sf_map
        sf_map.save('predictor/templates/predictor/sf_map.html')
        return render(request, 'predictor/index.html')

    return render(request, 'predictor/index.html')
