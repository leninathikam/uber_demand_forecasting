from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

from app import flatten_transform_output
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
scaler = joblib.load(asset_paths["scaler"])
kmeans = joblib.load(asset_paths["kmeans"])

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


def test_neighborhood_distance_conversion_supports_pandas_output():
    plot_data = pd.read_csv(asset_paths["plot_data"])
    sample_loc = plot_data.sample(1, random_state=42).reset_index(drop=True)
    scaled_coord = scaler.transform(sample_loc.iloc[:, 0:2])
    distances = flatten_transform_output(kmeans.transform(scaled_coord))

    assert isinstance(distances, list)
    assert len(distances) == 30
    assert all(isinstance(value, (float, np.floating)) for value in distances)
