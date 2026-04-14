from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_EXTERNAL_DIR = ROOT_PATH / "data" / "external"
DATA_PROCESSED_DIR = ROOT_PATH / "data" / "processed"
MODELS_DIR = ROOT_PATH / "models"

PLOT_DATA_PATH = DATA_EXTERNAL_DIR / "plot_data.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.csv"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ENCODER_PATH = MODELS_DIR / "encoder.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"
KMEANS_PATH = MODELS_DIR / "mb_kmeans.joblib"
RUN_INFO_PATH = ROOT_PATH / "run_information.json"


def ensure_local_assets(force: bool = False) -> dict[str, Path]:
    required_paths = [
        PLOT_DATA_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        MODEL_PATH,
        KMEANS_PATH,
        RUN_INFO_PATH,
    ]

    if not force and all(path.exists() for path in required_paths):
        return _path_mapping()

    DATA_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    plot_data = _build_plot_data(rng)

    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(
        plot_data.loc[:, ["pickup_latitude", "pickup_longitude"]]
    )

    kmeans = MiniBatchKMeans(
        n_clusters=30,
        n_init=10,
        random_state=42,
        batch_size=4096,
    )
    kmeans.fit(scaled_coords)

    plot_data = plot_data.copy()
    plot_data["region"] = kmeans.predict(scaled_coords)
    plot_data.to_csv(PLOT_DATA_PATH, index=False)

    train_df, test_df = _build_training_data(rng)
    train_df.to_csv(TRAIN_DATA_PATH, index=True)
    test_df.to_csv(TEST_DATA_PATH, index=True)

    encoder = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                ["region", "day_of_week"],
            )
        ],
        remainder="passthrough",
    )

    X_train = train_df.drop(columns=["total_pickups"])
    y_train = train_df["total_pickups"]
    encoder.fit(X_train)
    X_train_encoded = pd.DataFrame(
        encoder.transform(X_train),
        columns=encoder.get_feature_names_out(),
        index=X_train.index,
    )

    model = LinearRegression()
    model.fit(X_train_encoded, y_train)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(model, MODEL_PATH)

    run_information = {
        "mode": "local",
        "model_uri": str(MODEL_PATH),
        "model_path": str(MODEL_PATH),
        "artifact_path": "models",
    }
    RUN_INFO_PATH.write_text(json.dumps(run_information, indent=4), encoding="utf-8")

    return _path_mapping()


def _path_mapping() -> dict[str, Path]:
    return {
        "plot_data": PLOT_DATA_PATH,
        "train_data": TRAIN_DATA_PATH,
        "test_data": TEST_DATA_PATH,
        "scaler": SCALER_PATH,
        "encoder": ENCODER_PATH,
        "model": MODEL_PATH,
        "kmeans": KMEANS_PATH,
        "run_info": RUN_INFO_PATH,
    }


def _build_plot_data(rng: np.random.Generator) -> pd.DataFrame:
    latitudes = np.linspace(40.62, 40.82, 6)
    longitudes = np.linspace(-74.00, -73.76, 5)
    centers = np.array([(lat, lon) for lat in latitudes for lon in longitudes])

    rows: list[dict[str, float]] = []
    for lat, lon in centers:
        for _ in range(8):
            rows.append(
                {
                    "pickup_latitude": float(lat + rng.normal(0, 0.004)),
                    "pickup_longitude": float(lon + rng.normal(0, 0.004)),
                }
            )

    return pd.DataFrame(rows)


def _build_training_data(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range(
        "2016-01-01 00:00:00",
        "2016-03-31 23:45:00",
        freq="15min",
    )
    region_ids = np.arange(30)
    region_bias = np.linspace(18, 72, num=30)

    frames: list[pd.DataFrame] = []
    for region, bias in zip(region_ids, region_bias):
        hours = timestamps.hour + (timestamps.minute / 60.0)
        day_of_week = timestamps.dayofweek
        month = timestamps.month
        morning_peak = 14 * np.exp(-0.5 * ((hours - 8.0) / 1.8) ** 2)
        evening_peak = 18 * np.exp(-0.5 * ((hours - 18.5) / 2.3) ** 2)
        late_night = 6 * np.exp(-0.5 * ((hours - 1.0) / 2.0) ** 2)
        weekend_boost = np.where(day_of_week >= 4, 7, 0)
        month_trend = np.choose(month - 1, [0, 2, 4])
        noise = rng.normal(0, 1.2, size=len(timestamps))

        total_pickups = (
            bias
            + morning_peak
            + evening_peak
            + late_night
            + weekend_boost
            + month_trend
            + noise
        )
        total_pickups = np.clip(np.rint(total_pickups), 10, None).astype(int)

        frame = pd.DataFrame(
            {
                "tpep_pickup_datetime": timestamps,
                "region": region,
                "total_pickups": total_pickups,
                "day_of_week": day_of_week,
                "month": month,
            }
        )
        frame["avg_pickups"] = (
            frame["total_pickups"]
            .ewm(alpha=0.4, adjust=False)
            .mean()
            .round()
            .astype(int)
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["region", "tpep_pickup_datetime"]).reset_index(drop=True)

    for lag in range(1, 5):
        data[f"lag_{lag}"] = data.groupby("region")["total_pickups"].shift(lag)

    data = data.dropna().copy()
    data["lag_1"] = data["lag_1"].astype(int)
    data["lag_2"] = data["lag_2"].astype(int)
    data["lag_3"] = data["lag_3"].astype(int)
    data["lag_4"] = data["lag_4"].astype(int)

    ordered_columns = [
        "tpep_pickup_datetime",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "region",
        "total_pickups",
        "avg_pickups",
        "day_of_week",
        "month",
    ]
    data = data.loc[:, ordered_columns]
    data = data.set_index("tpep_pickup_datetime")

    train_df = data.loc[data["month"].isin([1, 2]), "lag_1":"day_of_week"]
    test_df = data.loc[data["month"] == 3, "lag_1":"day_of_week"]
    return train_df, test_df
