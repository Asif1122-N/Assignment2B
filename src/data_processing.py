"""This file prepares the SCATS traffic data for machine learning.

The original data has 96 traffic readings per day, based on 15-minute intervals.
This code changes the data into a time-series format so the ML models can learn
traffic patterns and predict future traffic flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SequenceData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_meta: pd.DataFrame
    test_meta: pd.DataFrame
    feature_cols: list[str]
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


def load_time_series(path: str | Path) -> pd.DataFrame:
    """Load scats_time_series.csv and create Datetime, Hour and day features."""
    df = pd.read_csv(path)

    # version one of a time column - some rows have "Time", others have "TimeSlot" with format "T_00:15".
    if "Time" not in df.columns:
        df["Time"] = df["TimeSlot"].astype(str).str.split("_").str[-1]

    df["SCATS Number"] = df["SCATS Number"].astype(int)
    df["Traffic"] = pd.to_numeric(df["Traffic"], errors="coerce")
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(subset=["Datetime", "Traffic"])
    df = df.groupby(["SCATS Number", "Datetime"], as_index=False)["Traffic"].mean()
    df = df.sort_values(["SCATS Number", "Datetime"]).reset_index(drop=True)

    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Cyclical encoding helps ML models understand time wraps around daily/weekly.
    quarter_of_day = df["Hour"] * 4 + df["Minute"] // 15
    df["TimeSin"] = np.sin(2 * np.pi * quarter_of_day / 96)
    df["TimeCos"] = np.cos(2 * np.pi * quarter_of_day / 96)
    df["DaySin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    return df


def make_tabular_features(df: pd.DataFrame, lags: Iterable[int] = (1, 2, 4, 8, 96)) -> tuple[pd.DataFrame, list[str]]:
    """Create lag/rolling features for a classical ML model such as Random Forest."""
    data = df.copy().sort_values(["SCATS Number", "Datetime"])

    for lag in lags:
        data[f"Lag{lag}"] = data.groupby("SCATS Number")["Traffic"].shift(lag)

    data["RollingMean4"] = (
        data.groupby("SCATS Number")["Traffic"]
        .shift(1)
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data["RollingMean8"] = (
        data.groupby("SCATS Number")["Traffic"]
        .shift(1)
        .rolling(8)
        .mean()
        .reset_index(level=0, drop=True)
    )

    feature_cols = [
        "SCATS Number",
        "Hour",
        "Minute",
        "DayOfWeek",
        "IsWeekend",
        "TimeSin",
        "TimeCos",
        "DaySin",
        "DayCos",
        *[f"Lag{lag}" for lag in lags],
        "RollingMean4",
        "RollingMean8",
    ]
    data = data.dropna(subset=feature_cols + ["Traffic"]).reset_index(drop=True)
    return data, feature_cols


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each SCATS site's data chronologically to avoid future data leakage."""
    train_parts = []
    test_parts = []
    for _, site_df in df.groupby("SCATS Number", sort=False):
        site_df = site_df.sort_values("Datetime")
        split_idx = int(len(site_df) * (1 - test_size))
        train_parts.append(site_df.iloc[:split_idx])
        test_parts.append(site_df.iloc[split_idx:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def make_sequence_data(df: pd.DataFrame, lookback: int = 8, test_size: float = 0.2) -> SequenceData:
    """Create scaled 3D sequence arrays for LSTM/GRU.

    X shape: (samples, lookback, features)
    y shape: (samples, 1)
    """
    feature_cols = ["Traffic", "TimeSin", "TimeCos", "DaySin", "DayCos", "IsWeekend"]
    sequences = []
    targets = []
    meta_rows = []

    for site, site_df in df.groupby("SCATS Number"):
        site_df = site_df.sort_values("Datetime").reset_index(drop=True)
        values = site_df[feature_cols].to_numpy(dtype="float32")
        traffic = site_df[["Traffic"]].to_numpy(dtype="float32")
        for i in range(lookback, len(site_df)):
            sequences.append(values[i - lookback : i])
            targets.append(traffic[i])
            meta_rows.append({
                "SCATS Number": int(site),
                "Datetime": site_df.loc[i, "Datetime"],
                "ActualTraffic": float(traffic[i][0]),
            })

    X = np.asarray(sequences, dtype="float32")
    y = np.asarray(targets, dtype="float32")
    meta = pd.DataFrame(meta_rows)

    train_idx = []
    test_idx = []
    for _, idxs in meta.groupby("SCATS Number").groups.items():
        idxs = list(idxs)
        split_idx = int(len(idxs) * (1 - test_size))
        train_idx.extend(idxs[:split_idx])
        test_idx.extend(idxs[split_idx:])

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    feature_scaler.fit(X_train_2d)
    target_scaler.fit(y_train_raw)

    X_train = feature_scaler.transform(X_train_2d).reshape(X_train_raw.shape)
    X_test = feature_scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape)
    y_train = target_scaler.transform(y_train_raw)
    y_test = target_scaler.transform(y_test_raw)

    return SequenceData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_meta=meta.iloc[train_idx].reset_index(drop=True),
        test_meta=meta.iloc[test_idx].reset_index(drop=True),
        feature_cols=feature_cols,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


def get_site_locations(cleaned_csv_path: str | Path) -> pd.DataFrame:
    """Return one latitude/longitude row per SCATS site from cleaned_scats_data.csv."""
    df = pd.read_csv(cleaned_csv_path)
    site_cols = ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]
    locations = df[site_cols].drop_duplicates("SCATS Number").copy()
    locations["SCATS Number"] = locations["SCATS Number"].astype(int)
    locations["NB_LATITUDE"] = pd.to_numeric(locations["NB_LATITUDE"], errors="coerce")
    locations["NB_LONGITUDE"] = pd.to_numeric(locations["NB_LONGITUDE"], errors="coerce")
    return locations.dropna(subset=["NB_LATITUDE", "NB_LONGITUDE"]).reset_index(drop=True)
