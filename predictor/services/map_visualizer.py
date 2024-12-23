import folium
import pandas as pd


class MapVisualizer:
    """
    handles map generation and marker management.
    """
    def __init__(self, location=[37.79016837, -122.415677], zoom_start=14):
        self.map = folium.Map(
            location=location,
            zoom_start=zoom_start,
            tiles='cartodbpositron',
            width='90%',
            height='400px'
        )

    def add_markers(self, predictions: pd.DataFrame):
        """
        params:
        predictions: DataFrame containing prediction results with latitude, longitude, and label.
        """
        for _, row in predictions.iterrows():
            label = row['prediction_label']
            color = 'red' if label > 0.85 else 'orange' if label > 0.6 else 'green'

            folium.Marker(
                location=[row['lat'], row['lng']],
                tooltip=row['STREET_BLOCK'],
                popup=round(label, 2),
                icon=folium.Icon(icon='fa-car', prefix='fa', color=color)
            ).add_to(self.map)

    def save_map(self, file_path):
        self.map.save(file_path)

    def render_map(self):
        """
        returns the map's HTML as a string using get_root()._repr_html_().
        """
        if self.map is None:
            raise ValueError("Map has not been initialized. Call 'initialize_map' first.")
        return self.map.get_root()._repr_html_()
