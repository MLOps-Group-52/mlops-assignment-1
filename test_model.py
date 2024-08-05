import pytest
import pandas as pd
import os
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


def test_train_and_save_model(path: pytest.MonkeyPatch):
    def mock_load_taxifare_data():
        return sample_data
    path.setattr('model_training.load_taxifare_data', mock_load_taxifare_data)
    train_and_save_model()
    assert os.path.exists('model.joblib')
    import joblib
    model = joblib.load('model.joblib')
    assert model is not None


def test_linear_model(sample_data: pd.DataFrame):
    train_size = 0.8
    test_size = 0.2
    target_feature = 'total_fare'
    continuous_features = [
        'trip_duration', 'distance_traveled',
        'fare', 'tip', 'miscellaneous_fees'
    ]
    r2, mse = LinearModel(
        train_size, test_size, sample_data, target_feature, continuous_features
    )
    assert isinstance(r2, float)
    assert isinstance(mse, float)
    assert 0 <= r2 <= 1
    assert mse >= 0


if __name__ == "__main__":
    pytest.main()
