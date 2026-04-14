import datetime as dt
from time import sleep

import joblib
import pandas as pd
import streamlit as st
from sklearn import set_config
from sklearn.pipeline import Pipeline

from src.local_assets import ensure_local_assets


set_config(transform_output="pandas")

asset_paths = ensure_local_assets()

scaler = joblib.load(asset_paths["scaler"])
encoder = joblib.load(asset_paths["encoder"])
model = joblib.load(asset_paths["model"])
kmeans = joblib.load(asset_paths["kmeans"])

df_plot = pd.read_csv(asset_paths["plot_data"])
df = pd.read_csv(
    asset_paths["test_data"],
    parse_dates=["tpep_pickup_datetime"],
).set_index("tpep_pickup_datetime")

st.title("Uber Demand in New York City")
st.caption("Running fully locally with generated sample assets. No cloud services or remote registry are required.")

st.sidebar.title("Options")
map_type = st.sidebar.radio(
    label="Select the type of Map",
    options=["Complete NYC Map", "Only for Neighborhood Regions"],
    index=1,
)

st.subheader("Date")
date = st.date_input(
    "Select the date",
    value=None,
    min_value=dt.date(year=2016, month=3, day=1),
    max_value=dt.date(year=2016, month=3, day=31),
)
st.write("**Date:**", date)

st.subheader("Time")
time = st.time_input("Select the time", value=None)
st.write("**Current Time:**", time)

if date and time:
    delta = dt.timedelta(minutes=15)
    next_interval = dt.datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=time.hour,
        minute=time.minute,
    ) + delta
    st.write("Demand for Time: ", next_interval.time())

    index = pd.Timestamp(f"{date} {next_interval.time()}")
    st.write("**Date & Time:**", index)

    st.subheader("Location")
    sample_loc = df_plot.sample(1, random_state=42).reset_index(drop=True)
    lat = sample_loc["pickup_latitude"].item()
    long = sample_loc["pickup_longitude"].item()
    region = int(sample_loc["region"].item())
    st.write("**Your Current Location**")
    st.write(f"Lat: {lat}")
    st.write(f"Long: {long}")

    with st.spinner("Fetching your Current Region"):
        sleep(1)

    st.write("Region ID: ", region)
    scaled_coord = scaler.transform(sample_loc.iloc[:, 0:2])

    st.subheader("MAP")
    colors = [
        "#FF0000",
        "#FF4500",
        "#FF8C00",
        "#FFD700",
        "#ADFF2F",
        "#32CD32",
        "#008000",
        "#006400",
        "#00FF00",
        "#7CFC00",
        "#00FA9A",
        "#00FFFF",
        "#40E0D0",
        "#4682B4",
        "#1E90FF",
        "#0000FF",
        "#0000CD",
        "#8A2BE2",
        "#9932CC",
        "#BA55D3",
        "#FF00FF",
        "#FF1493",
        "#C71585",
        "#FF4500",
        "#FF6347",
        "#FFA07A",
        "#FFDAB9",
        "#FFE4B5",
        "#F5DEB3",
        "#EEE8AA",
    ]

    region_colors = {
        region_id: colors[i]
        for i, region_id in enumerate(sorted(df_plot["region"].unique().tolist()))
    }
    df_plot = df_plot.copy()
    df_plot["color"] = df_plot["region"].map(region_colors)

    pipe = Pipeline(
        [
            ("encoder", encoder),
            ("reg", model),
        ]
    )

    if map_type == "Complete NYC Map":
        progress_bar = st.progress(value=0, text="Operation in progress. Please wait.")
        for percent_complete in range(100):
            sleep(0.01)
            progress_bar.progress(percent_complete + 1, text="Operation in progress. Please wait.")

        st.map(
            data=df_plot,
            latitude="pickup_latitude",
            longitude="pickup_longitude",
            size=0.01,
            color="color",
        )
        progress_bar.empty()

        input_data = df.loc[index, :].sort_values("region")
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Map Legend")
        for ind in range(30):
            color = colors[ind]
            demand = predictions[ind]
            region_id = f"{ind} (Current region)" if region == ind else ind
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f"Region ID: {region_id} <br>"
                f"Demand: {int(demand)} <br> <br>",
                unsafe_allow_html=True,
            )

    elif map_type == "Only for Neighborhood Regions":
        distances = kmeans.transform(scaled_coord).ravel().tolist()
        sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])[0:9]
        indexes = sorted([ind[0] for ind in sorted_distances])

        df_plot_filtered = df_plot[df_plot["region"].isin(indexes)]

        progress_bar = st.progress(value=0, text="Operation in progress. Please wait.")
        for percent_complete in range(100):
            sleep(0.01)
            progress_bar.progress(percent_complete + 1, text="Operation in progress. Please wait.")

        st.map(
            data=df_plot_filtered,
            latitude="pickup_latitude",
            longitude="pickup_longitude",
            size=0.01,
            color="color",
        )
        progress_bar.empty()

        input_data = df.loc[index, :]
        input_data = input_data.loc[input_data["region"].isin(indexes), :].sort_values("region")
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Map Legend")
        for ind in range(9):
            color = colors[indexes[ind]]
            demand = predictions[ind]
            region_id = (
                f"{indexes[ind]} (Current region)"
                if region == indexes[ind]
                else indexes[ind]
            )
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f"Region ID: {region_id} <br>"
                f"Demand: {int(demand)} <br> <br>",
                unsafe_allow_html=True,
            )
