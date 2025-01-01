from datetime import datetime

import pandas as pd

from predictor.exceptions import InternalProcessingError
from predictor.utils.loaders import load_data_file


class DataHandler:
    """
    handles predictions input data and map-related data
    """
    _REQUIRED_COLUMNS_BASIC_DATA = ['BLOCK_ID', 'STREET_BLOCK', 'lat', 'lng']
    _REQUIRED_COLUMNS_PREDICTIONS = ['BLOCK_ID', 'Dayofweek', 'Hour', 'holiday', 'Precipitation', 'DAY_TYPE']

    def __init__(self):
        self.basic_data = load_data_file()
        self._validate_columns(self.basic_data, self._REQUIRED_COLUMNS_BASIC_DATA, "Data file")


    def _validate_columns(self, df: pd.DataFrame, required_columns: list, context: str) -> None:
        """
        validates that the DataFrame contains the required columns
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise InternalProcessingError(
                f"{context} is missing required columns: {', '.join(missing_columns)}"
            )


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
        prediction_input = self.basic_data[['BLOCK_ID']].drop_duplicates().assign(
            Dayofweek=day_of_week,
            Hour=hour,
            holiday=holiday,
            Precipitation=rain,
            DAY_TYPE=0 if day_of_week <= 4 else 1
        )
        self._validate_columns(prediction_input, self._REQUIRED_COLUMNS_PREDICTIONS, "Prediction input data")
        return prediction_input

    def prepare_map_data(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        param:
        predictions: DataFrame containing BLOCK_ID and prediction_label
        """
        self._validate_columns(predictions, ['BLOCK_ID', 'prediction_label'], "Predictions")

        predictions_for_map = predictions[['BLOCK_ID', 'prediction_label']]
        return predictions_for_map.merge(self.basic_data, on='BLOCK_ID', how='left')
