from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

from src.local_assets import ensure_local_assets


set_config(transform_output="pandas")

current_path = Path(__file__)
root_path = current_path.parent.parent
asset_paths = ensure_local_assets()

train_data_path = asset_paths["train_data"]
test_data_path = asset_paths["test_data"]

encoder_path = asset_paths["encoder"]
encoder = joblib.load(encoder_path)
model = joblib.load(asset_paths["model"])

model_pipe = Pipeline(
    steps=[
        ("encoder", encoder),
        ("regressor", model),
    ]
)


@pytest.mark.parametrize(
    argnames="data_path,threshold",
    argvalues=[(train_data_path, 0.15), (test_data_path, 0.15)],
)
def test_performance(data_path, threshold):
    data = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index(
        "tpep_pickup_datetime"
    )
    X = data.drop(columns=["total_pickups"])
    y = data["total_pickups"]
    y_pred = model_pipe.predict(X)
    loss = mean_absolute_percentage_error(y, y_pred)
    assert loss <= threshold, f"The model does not pass the performance threshold of {threshold}"
