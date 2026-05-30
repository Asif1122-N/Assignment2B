"""Prepares SCATS traffic data for ML.

Loading the CVS time-series data, creating features for XGBoost, and building scaled 3D arrays for LSTM/GRU training. 
Also includes a helper to get site locations for routing.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SequenceData:
    """This is where the LSTM/GRU training data is stored."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_meta: pd.DataFrame
    test_meta: pd.DataFrame
    feature_cols: list
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


def load_time_series(path) -> pd.DataFrame:
    """Load scats_time_series.csv and add time/day features."""
    df = pd.read_csv(path)

    # Handle both "Time" and "TimeSlot" column formats
    if "Time" not in df.columns:
        df["Time"] = df["TimeSlot"].astype(str).str.split("_").str[-1]

    df["SCATS Number"] = df["SCATS Number"].astype(int)
    df["Traffic"] = pd.to_numeric(df["Traffic"], errors="coerce")
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    # NO DUPLICATES
    df = df.dropna(subset=["Datetime", "Traffic"])
    df = df.groupby(["SCATS Number", "Datetime"], as_index=False)["Traffic"].mean()
    df = df.sort_values(["SCATS Number", "Datetime"]).reset_index(drop=True)

    # Time features (hour, minute, day of week)
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # THE important time features to capture cyclical patterns in time and day of week
    df["TimeSin"] = np.sin(2 * np.pi * slot / 96)
    df["TimeCos"] = np.cos(2 * np.pi * slot / 96)
    df["DaySin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    return df


def make_tabular_features(df: pd.DataFrame, lags=(1, 2, 4, 8, 96)):
    """Add lag and rolling mean features for XGBoost."""
    data = df.copy().sort_values(["SCATS Number", "Datetime"])

    # Lag features — previous traffic readings per site
    for lag in lags:
        data[f"Lag{lag}"] = data.groupby("SCATS Number")["Traffic"].shift(lag)

    #Creating a rolling mean of the 4 and 8 readings before results in better performance for XGBoost, as it captures short-term trends and smooths out noise in the traffic data. 
    # This can help the model learn more stable patterns and improve its predictive accuracy.
    grouped = data.groupby("SCATS Number")["Traffic"].shift(1)
    data["RollingMean4"] = grouped.rolling(4).mean().reset_index(level=0, drop=True)
    data["RollingMean8"] = grouped.rolling(8).mean().reset_index(level=0, drop=True)

    feature_cols = [
        "SCATS Number", "Hour", "Minute", "DayOfWeek", "IsWeekend",
        "TimeSin", "TimeCos", "DaySin", "DayCos",
        *[f"Lag{lag}" for lag in lags],
        "RollingMean4", "RollingMean8",
    ]

    data = data.dropna(subset=feature_cols + ["Traffic"]).reset_index(drop=True)
    return data, feature_cols


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split each site's data chronologically — no future data leakage."""
    train_parts, test_parts = [], []

    for _, site_df in df.groupby("SCATS Number", sort=False):
        site_df = site_df.sort_values("Datetime")
        split = int(len(site_df) * (1 - test_size))
        train_parts.append(site_df.iloc[:split])
        test_parts.append(site_df.iloc[split:])

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def make_sequence_data(df: pd.DataFrame, lookback: int = 8, test_size: float = 0.2) -> SequenceData:
    """Build scaled 3D arrays for LSTM/GRU.

    X shape: (samples, lookback, features)
    y shape: (samples, 1)
    """
    feature_cols = ["Traffic", "TimeSin", "TimeCos", "DaySin", "DayCos", "IsWeekend"]
    sequences, targets, meta_rows = [], [], []

    for site, site_df in df.groupby("SCATS Number"):
        site_df = site_df.sort_values("Datetime").reset_index(drop=True)
        values = site_df[feature_cols].to_numpy(dtype="float32")
        traffic = site_df[["Traffic"]].to_numpy(dtype="float32")

        for i in range(lookback, len(site_df)):
            sequences.append(values[i - lookback: i])
            targets.append(traffic[i])
            meta_rows.append({
                "SCATS Number": int(site),
                "Datetime": site_df.loc[i, "Datetime"],
                "ActualTraffic": float(traffic[i][0]),
            })

    X = np.asarray(sequences, dtype="float32")
    y = np.asarray(targets, dtype="float32")
    meta = pd.DataFrame(meta_rows)

    # Diving the data into test and train sets based on SCATS Number to avoid data leakage between sites.
    train_idx, test_idx = [], []
    for idxs in meta.groupby("SCATS Number").groups.values():
        idxs = list(idxs)
        split = int(len(idxs) * (1 - test_size))
        train_idx.extend(idxs[:split])
        test_idx.extend(idxs[split:])

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]

    #Scale the features and target to [0, 1] range for better LSTM/GRU training stability
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    n_features = X_train_raw.shape[-1]
    X_train = feature_scaler.fit_transform(
        X_train_raw.reshape(-1, n_features)
    ).reshape(X_train_raw.shape)
    X_test = feature_scaler.transform(
        X_test_raw.reshape(-1, n_features)
    ).reshape(X_test_raw.shape)
    y_train = target_scaler.fit_transform(y_train_raw)
    y_test = target_scaler.transform(y_test_raw)

    return SequenceData(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        train_meta=meta.iloc[train_idx].reset_index(drop=True),
        test_meta=meta.iloc[test_idx].reset_index(drop=True),
        feature_cols=feature_cols,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


def get_site_locations(cleaned_csv_path) -> pd.DataFrame:
    """Return one lat/lon row per SCATS site."""
    df = pd.read_csv(cleaned_csv_path)
    cols = ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]
    locations = df[cols].drop_duplicates("SCATS Number").copy()
    locations["SCATS Number"] = locations["SCATS Number"].astype(int)
    locations["NB_LATITUDE"] = pd.to_numeric(locations["NB_LATITUDE"], errors="coerce")
    locations["NB_LONGITUDE"] = pd.to_numeric(locations["NB_LONGITUDE"], errors="coerce")
    return locations.dropna(subset=["NB_LATITUDE", "NB_LONGITUDE"]).reset_index(drop=True)
