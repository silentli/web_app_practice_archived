import pandas as pd
from typing import Optional
from pycaret.regression import predict_model

from ..utils.loaders import load_prediction_model


class Predictor:
    def __init__(self, model_name: Optional[str] = None):
        self.model = load_prediction_model(model_name)


    def predict(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        predictions = predict_model(self.model, data=preprocessed_data)
        return predictions
