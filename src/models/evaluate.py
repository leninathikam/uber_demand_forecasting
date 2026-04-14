import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error

from src.local_assets import ensure_local_assets


set_config(transform_output="pandas")

logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)


def load_model(model_path):
    return joblib.load(model_path)


def save_run_information(metrics, model_uri, path):
    run_information = {
        "mode": "local",
        "metrics": metrics,
        "artifact_path": "models",
        "model_uri": str(model_uri),
    }
    path.write_text(json.dumps(run_information, indent=4), encoding="utf-8")


if __name__ == "__main__":
    ensure_local_assets()
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    train_data_path = root_path / "data" / "processed" / "train.csv"
    test_data_path = root_path / "data" / "processed" / "test.csv"

    df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")

    df.set_index("tpep_pickup_datetime", inplace=True)
    X_test = df.drop(columns=["total_pickups"])
    y_test = df["total_pickups"]

    encoder_path = root_path / "models" / "encoder.joblib"
    encoder = joblib.load(encoder_path)
    logger.info("Encoder loaded successfully")

    X_test_encoded = encoder.transform(X_test)
    logger.info("Data transformed successfully")

    model_path = root_path / "models" / "model.joblib"
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    y_pred = model.predict(X_test_encoded)
    loss = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Loss: {loss}")

    json_file_save_path = root_path / "run_information.json"
    save_run_information(
        metrics={"MAPE": float(loss)},
        model_uri=model_path,
        path=json_file_save_path,
    )
    logger.info("Run information saved successfully")
