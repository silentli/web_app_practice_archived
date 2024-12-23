import os
import pandas as pd
from django.conf import settings
from pycaret.regression import load_model

def load_prediction_model(model_name='park_pycaret_2012_pipeline'):
    model_path = os.path.join(settings.BASE_DIR, 'predictor', 'model', model_name)
    return load_model(model_path)

def load_data_file():
    data_path = os.path.join(settings.BASE_DIR, 'predictor', 'data', 'street_sensor_longlat.csv')
    return pd.read_csv(data_path)
