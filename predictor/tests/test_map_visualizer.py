import pytest
import pandas as pd
import folium

from predictor.services.map_visualizer import MapVisualizer

@pytest.fixture
def mock_predictions():
   return pd.DataFrame({
       'BLOCK_ID': [41522, 21733],
       'STREET_BLOCK': ['FILLMORE ST 2200', '17TH ST 3300'],
       'lat': [37.7900066, 37.763444],
       'lng': [-122.4339023, -122.4195514],
       'prediction_label': [0.87, 0.55],
   })

@pytest.fixture
def map_visualizer():
    return MapVisualizer()

def test_map_initialization(map_visualizer):
    assert isinstance(map_visualizer.map, folium.Map), "Map should be an instance of folium.Map."

def test_add_makers(map_visualizer, mock_predictions):
    map_visualizer.add_markers(mock_predictions)
    assert len(map_visualizer.map._children) >= len(mock_predictions), "Map should have markers added to it."

def test_save_map(tmp_path, map_visualizer, mock_predictions):
    map_visualizer.add_markers(mock_predictions)

    file_path = tmp_path / "test_map.html"
    map_visualizer.save_map(file_path)

    assert file_path.exists(), "The map file should be saved."
    assert file_path.stat().st_size > 0, "The map file should not be empty."
