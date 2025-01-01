from typing import Optional

import pandas as pd
from pycaret.regression import predict_model

from predictor.exceptions import InternalProcessingError
from predictor.utils.loaders import load_prediction_model


class Predictor:
    def __init__(self, model_name: Optional[str] = None):
        self.model = load_prediction_model(model_name)


    def predict(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        if preprocessed_data.empty:
            raise InternalProcessingError("Data for prediction is empty. Ensure preprocessing was successful.")
        predictions = predict_model(self.model, data=preprocessed_data)
        return predictions
