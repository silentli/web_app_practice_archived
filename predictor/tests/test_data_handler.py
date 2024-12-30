import pytest
import pandas as pd
from unittest.mock import patch

from predictor.services.data_handler import DataHandler

@pytest.fixture
@patch("predictor.services.data_handler.load_data_file")
def data_handler(mock_load_data_file):
    mock_data = pd.DataFrame({
       'BLOCK_ID': [41522, 21733],
       'STREET_BLOCK': ['FILLMORE ST 2200', '17TH ST 3300'],
       'lat': [37.7900066, 37.763444],
       'lng': [-122.4339023, -122.4195514],
   })
    mock_load_data_file.return_value = mock_data
    return DataHandler()


def test_prepare_prediction_data(data_handler):
    date = pd.Timestamp('2014-02-01')
    hour = 11
    holiday = 1
    rain = 0

    expected_day_type = 0 if date.weekday() <= 4 else 1
    input_data = data_handler.prepare_prediction_data(date, hour, holiday, rain)

    assert not input_data.empty, "Input data should not be empty."
    assert set(data_handler._REQUIRED_COLUMNS_PREDICTIONS) <= set(input_data.columns), \
        f"Input data columns: '{input_data.columns}' do not include all required columns."
    assert (input_data['Dayofweek'] == date.weekday()).all(), "Dayofweek column mismatch."
    assert (input_data['Hour'] == hour).all(), "Hour column mismatch."
    assert (input_data['holiday'] == holiday).all(), "Holiday column mismatch."
    assert (input_data['DAY_TYPE'] == expected_day_type).all(), f"DAY_TYPE column mismatch."


def test_prepare_map_data(data_handler):
    predictions = pd.DataFrame({
        'BLOCK_ID': [1, 2, 3],
        'prediction_label': [0.5, 0.8, 0.6],
    })

    map_data = data_handler.prepare_map_data(predictions)

    assert not map_data.empty, "Map data should not be empty."
    assert 'BLOCK_ID' in map_data.columns, "BLOCK_ID column is missing in the map data."
    assert 'prediction_label' in map_data.columns, "prediction_label column is missing in map data."
    assert len(map_data) == len(predictions), "Map data row count should match predictions row count."
