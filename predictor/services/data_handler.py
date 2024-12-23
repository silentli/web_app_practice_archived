import pandas as pd
from datetime import datetime

from ..utils.loaders import load_data_file


class DataHandler:
    """
    handles predictions input data and map-related data
    """
    REQUIRED_COLUMNS = ['BLOCK_ID', 'Dayofweek', 'Hour', 'holiday', 'Precipitation', 'DAY_TYPE']

    def __init__(self):
        self.basic_data = load_data_file()

    def prepare_prediction_data(self, date: datetime.date, hour: int, holiday: int, rain: int) -> pd.DataFrame:
        """
        params:
        date: Date in "dd/mm/yyyy" format.
        hour: Hour as an integer.
        holiday: Holiday indicator (1 = Yes, 0 = No).
        rain: Rain indicator (1 = Yes, 0 = No).
        """
        # day_of_week = datetime.strptime(date, '%d/%m/%Y').weekday()
        day_of_week = date.weekday()
        prediction_input = self.basic_data[['BLOCK_ID']].drop_duplicates()
        return prediction_input.assign(
            Dayofweek=day_of_week,
            Hour=hour,
            holiday=holiday,
            Precipitation=rain,
            DAY_TYPE=0 if day_of_week <= 4 else 1
        )

    def prepare_map_data(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        param:
        predictions: DataFrame containing BLOCK_ID and prediction_label
        """
        if 'BLOCK_ID' not in predictions.columns or 'prediction_label' not in predictions.columns:
            raise ValueError("Predictions must include 'BLOCK_ID' and 'prediction_label' columns.")

        predictions_for_map = predictions[['BLOCK_ID', 'prediction_label']]
        return predictions_for_map.merge(self.basic_data, on='BLOCK_ID', how='left')
