from pathlib import Path
import sys

import joblib
import pandas as pd


# 1. Set project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"

sys.path.append(str(SRC_DIR))

from config import TIME_SERIES_DATA
from data_processing import load_time_series, make_tabular_features


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


# 4. Predict traffic flow using XGBoost

def predict_xgboost_flow(scats_number, datetime_value):
    """
    Predict traffic flow using the saved XGBoost model.

    This uses the same dataset and feature generation method used during training.
    The model predicts traffic for one 15-minute interval.
    The returned value is vehicles per hour.
    """

    datetime_value = pd.to_datetime(datetime_value)

    matched_row = feature_df[
        (feature_df["SCATS Number"] == scats_number)
        & (feature_df["Datetime"] == datetime_value)
    ]

    if matched_row.empty:
        raise ValueError(
            f"No matching row found for SCATS {scats_number} at {datetime_value}. "
            "Use a date/time that exists in data/scats_time_series.csv, "
            "for example: 2006-10-22 09:00"
        )

    X = matched_row[model_feature_cols]

    predicted_15_min = model.predict(X)[0]
    predicted_per_hour = predicted_15_min * 4

    return float(predicted_per_hour), float(predicted_15_min)


# 5. User input test

if __name__ == "__main__":
    print("\nXGBoost Traffic Flow Prediction")
    print("--------------------------------")
    print("Example SCATS number: 3001")
    print("Example date and time: 2006-10-22 09:00")
    print("Format must be: YYYY-MM-DD HH:MM")
    print("--------------------------------")

    scats_number = int(input("Enter SCATS number: "))
    datetime_value = input("Enter date and time: ")

    try:
        predicted_per_hour, predicted_15_min = predict_xgboost_flow(
            scats_number=scats_number,
            datetime_value=datetime_value,
        )

        print("\nPrediction Result")
        print("-----------------")
        print("SCATS number:", scats_number)
        print("Date and time:", datetime_value)
        print("Predicted traffic in 15 min:", round(predicted_15_min, 2), "vehicles")
        print("Predicted flow per hour:", round(predicted_per_hour, 2), "vehicles/hour")

    except Exception as error:
        print("\nError:", error)