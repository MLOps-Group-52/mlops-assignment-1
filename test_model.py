import pytest
import pandas as pd
import joblib
from model_training import load_taxifare_data
from model_training import train_and_save_model
from model_training import LinearModel


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            'trip_duration': [10, 20, 30, 40, 50],
            'distance_traveled': [1, 2, 3, 4, 5],
            'fare': [5, 10, 15, 20, 25],
            'tip': [1, 2, 3, 4, 5],
            'miscellaneous_fees': [0.5, 1, 1.5, 2, 2.5],
            'total_fare': [6.5, 12, 18.5, 24, 30],
            'num_of_passengers': [1, 2, 1, 2, 1],
            'surge_applied': [0, 1, 0, 1, 0],
        }
    )


def test_load_taxifare_data():
    data = load_taxifare_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_train_and_save_model(monkeypatch, sample_data):
    def mock():
        return sample_data
    monkeypatch.setattr('model_training.load_taxifare_data', mock)
    train_and_save_model()
    # Load the model and check if it's not None
    model = joblib.load('Models/model.joblib')
    assert model is not None


if __name__ == "__main__":
    pytest.main()
