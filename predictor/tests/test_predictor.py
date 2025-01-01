from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictor.exceptions import InternalProcessingError
from predictor.services.predictor import Predictor


@pytest.fixture
def mock_loaded_model(monkeypatch):
    """mock the model loading function to return a dummy model"""
    mock_model = MagicMock()
    monkeypatch.setattr("predictor.services.predictor.load_prediction_model", lambda _: mock_model)
    return mock_model

@pytest.fixture
def mock_pycaret_predict(monkeypatch):
    """mock the PyCaret predict_model function"""
    mock_predict = MagicMock(return_value=pd.DataFrame({'prediction_label': [0.5, 0.7]}))
    monkeypatch.setattr("predictor.services.predictor.predict_model", mock_predict)
    return mock_predict

@pytest.fixture
def predictor(mock_loaded_model):
    """create a Predictor instance"""
    return Predictor(model_name="dummy_model")


def test_predictor_with_valid_data(predictor, mock_pycaret_predict):
    test_data = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    result = predictor.predict(test_data)

    assert not result.empty, "The output should not be empty."
    assert 'prediction_label' in result.columns, "The output must include the 'prediction_label' column."
    assert len(result) == len(test_data), "The output must have the same number of rows as the input."

    mock_pycaret_predict.assert_called_once_with(predictor.model, data=test_data)

def test_predictor_with_empty_data(predictor, mock_pycaret_predict):
    empty_data = pd.DataFrame(columns=['feature1', 'feature2'])
    with pytest.raises(InternalProcessingError):
        predictor.predict(empty_data)
