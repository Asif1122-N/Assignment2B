from pathlib import Path
import sys

import joblib
import pandas as pd


# 1. Set project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
LOCATION_DATA_PATH = PROJECT_ROOT / "data" / "ModifiedScatsDataOctober2006.csv"

sys.path.append(str(SRC_DIR))

from config import TIME_SERIES_DATA
from data_processing import load_time_series, make_tabular_features
from travel_time import edge_travel_time_minutes, haversine_km


# 2. Load XGBoost model

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
model_feature_cols = saved_data["feature_cols"]


# 3. Load same dataset used in training

raw_df = load_time_series(TIME_SERIES_DATA)
feature_df, feature_cols = make_tabular_features(raw_df)

feature_df["Datetime"] = pd.to_datetime(feature_df["Datetime"], errors="coerce")
feature_df = feature_df.dropna(subset=["Datetime"])
feature_df = feature_df.sort_values(["SCATS Number", "Datetime"])


# 4. Load original SCATS dataset for coordinates

location_df = pd.read_csv(
    LOCATION_DATA_PATH,
    skiprows=1
)

location_df.columns = location_df.columns.str.strip()

location_df["SCATS Number"] = pd.to_numeric(
    location_df["SCATS Number"],
    errors="coerce"
)

location_df["NB_LATITUDE"] = pd.to_numeric(
    location_df["NB_LATITUDE"],
    errors="coerce"
)

location_df["NB_LONGITUDE"] = pd.to_numeric(
    location_df["NB_LONGITUDE"],
    errors="coerce"
)

location_df = location_df.dropna(
    subset=["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]
)

location_df["SCATS Number"] = location_df["SCATS Number"].astype(int)


# 5. Get SCATS coordinates

def get_scats_coordinates(scats_number):
    site_rows = location_df[
        location_df["SCATS Number"] == scats_number
    ]

    if site_rows.empty:
        raise ValueError(
            f"SCATS number {scats_number} not found in location dataset."
        )

    lat = site_rows.iloc[0]["NB_LATITUDE"]
    lon = site_rows.iloc[0]["NB_LONGITUDE"]

    return float(lat), float(lon)


# 6. Predict traffic flow using XGBoost

def predict_xgboost_flow(scats_number, datetime_value):
    """
    Predict traffic flow using the saved XGBoost model.

    Uses the same dataset and feature generation method used during training.
    The model predicts traffic for one 15-minute interval.
    Returns both 15-minute traffic and vehicles per hour.
    """

    datetime_value = pd.to_datetime(datetime_value)

    matched_row = feature_df[
        (feature_df["SCATS Number"] == scats_number)
        & (feature_df["Datetime"] == datetime_value)
    ]

    if matched_row.empty:
        raise ValueError(
            f"No matching row found for SCATS {scats_number} at {datetime_value}. "
            "Use a date/time that exists in data/scats_time_series.csv. "
            "Example: 2006-10-22 09:00"
        )

    X = matched_row[model_feature_cols]

    predicted_15_min = float(model.predict(X)[0])
    predicted_per_hour = predicted_15_min * 4

    return predicted_15_min, predicted_per_hour


# 7. Convert XGBoost prediction into travel time

def predict_travel_time_between_scats(origin_scats, destination_scats, datetime_value):
    """
    Predict travel time between two SCATS sites.

    Uses:
    - coordinates from ModifiedScatsDataOctober2006.csv
    - XGBoost predicted traffic flow from origin SCATS
    - haversine distance between origin and destination
    - travel_time.py conversion from flow to time
    """

    origin_lat, origin_lon = get_scats_coordinates(origin_scats)
    destination_lat, destination_lon = get_scats_coordinates(destination_scats)

    distance_km = haversine_km(
        origin_lat,
        origin_lon,
        destination_lat,
        destination_lon,
    )

    predicted_15_min, predicted_per_hour = predict_xgboost_flow(
        origin_scats,
        datetime_value,
    )

    estimated_minutes = edge_travel_time_minutes(
        distance_km=distance_km,
        predicted_flow_per_hour=predicted_per_hour,
    )

    return {
        "origin_scats": origin_scats,
        "destination_scats": destination_scats,
        "datetime": str(datetime_value),
        "distance_km": distance_km,
        "predicted_traffic_15_min": predicted_15_min,
        "predicted_flow_per_hour": predicted_per_hour,
        "estimated_minutes": estimated_minutes,
    }


# 8. User input test

if __name__ == "__main__":
    print("\nXGBoost Traffic to Travel Time Prediction")
    print("----------------------------------------")
    print("Example origin SCATS: 3001")
    print("Example destination SCATS: 3002")
    print("Example date and time: 2006-10-22 09:00")
    print("Format must be: YYYY-MM-DD HH:MM")
    print("----------------------------------------")

    try:
        origin_scats = int(input("Enter origin SCATS number: "))
        destination_scats = int(input("Enter destination SCATS number: "))
        datetime_value = input("Enter date and time: ")

        result = predict_travel_time_between_scats(
            origin_scats=origin_scats,
            destination_scats=destination_scats,
            datetime_value=datetime_value,
        )

        print("\nPrediction Result")
        print("-----------------")
        print("Origin SCATS:", result["origin_scats"])
        print("Destination SCATS:", result["destination_scats"])
        print("Date and time:", result["datetime"])
        print("Distance:", round(result["distance_km"], 3), "km")
        print(
            "Predicted traffic in 15 min:",
            round(result["predicted_traffic_15_min"], 2),
            "vehicles",
        )
        print(
            "Predicted flow per hour:",
            round(result["predicted_flow_per_hour"], 2),
            "vehicles/hour",
        )
        print(
            "Estimated travel time:",
            round(result["estimated_minutes"], 2),
            "minutes",
        )

    except Exception as error:
        print("\nError:", error)