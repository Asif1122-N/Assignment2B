"""Train and test the three traffic prediction models: XGBoost, LSTM and GRU.

This file does three main things:
1.  Accesses the cleaned SCATS time-series data.
2. Trains the three ML models.
3. Saving the model results into the outputs folder for comparison.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    LOOKBACK_STEPS,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TIME_SERIES_DATA,
)
from data_processing import (
    load_time_series,
    make_sequence_data,
    make_tabular_features,
    temporal_train_test_split,
)


# 1. Model evaluation

def regression_metrics(y_true, y_pred) -> dict:
    """Calculate the accuracy scores used to compare the models."""

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    # MAPE is not divisible by 0, so we need to filter out zero values from y_true
    non_zero_mask = y_true_array != 0
    if non_zero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
                    / y_true_array[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "R2": r2,
    }


# ---------------------------------------------------------
# 2. XGBoost model
# ---------------------------------------------------------

def train_xgboost(df: pd.DataFrame) -> dict:
    """Train XGBoost using normal table-based traffic features."""

    from xgboost import XGBRegressor

    data, feature_cols = make_tabular_features(df)
    train_df, test_df = temporal_train_test_split(data, TEST_SIZE)

    X_train = train_df[feature_cols]
    y_train = train_df["Traffic"]

    X_test = test_df[feature_cols]
    y_test = test_df["Traffic"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    predictions = model.predict(X_test)

    metrics = regression_metrics(y_test, predictions)
    metrics["Model"] = "XGBoost"
    metrics["TrainingSeconds"] = train_seconds

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        MODEL_DIR / "xgboost.joblib",
    )

    save_predictions(
        test_df["SCATS Number"].values,
        test_df["Datetime"].values,
        y_test.values,
        predictions,
        "xgboost_predictions.csv",
    )

    return metrics



# 3. LSTM and GRU models


def build_deep_model(model_type: str, input_shape: tuple[int, int]):
    """Build either an LSTM or GRU model."""

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, GRU, Input, LSTM
    from tensorflow.keras.models import Sequential

    if model_type == "lstm":
        recurrent_layer = LSTM
    else:
        recurrent_layer = GRU

    model = Sequential(
        [
            Input(shape=input_shape),
            recurrent_layer(64),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    return model, early_stop


def train_deep_model(df: pd.DataFrame, model_type: str, epochs: int) -> dict:
    """Train either the LSTM or GRU model."""

    sequence_data = make_sequence_data(
        df,
        lookback=LOOKBACK_STEPS,
        test_size=TEST_SIZE,
    )

    model, early_stop = build_deep_model(
        model_type,
        input_shape=sequence_data.X_train.shape[1:],
    )

    start = time.time()
    history = model.fit(
        sequence_data.X_train,
        sequence_data.y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1,
    )
    train_seconds = time.time() - start

    predicted_scaled = model.predict(sequence_data.X_test).reshape(-1, 1)

    predictions = sequence_data.target_scaler.inverse_transform(
        predicted_scaled
    ).reshape(-1)

    actual = sequence_data.target_scaler.inverse_transform(
        sequence_data.y_test
    ).reshape(-1)

    metrics = regression_metrics(actual, predictions)
    metrics["Model"] = model_type.upper()
    metrics["TrainingSeconds"] = train_seconds

    model.save(MODEL_DIR / f"{model_type}_model.keras")

    joblib.dump(
        {
            "feature_scaler": sequence_data.feature_scaler,
            "target_scaler": sequence_data.target_scaler,
            "feature_cols": sequence_data.feature_cols,
            "lookback": LOOKBACK_STEPS,
        },
        MODEL_DIR / f"{model_type}_scalers.joblib",
    )

    prediction_df = sequence_data.test_meta.copy()
    prediction_df["Actual"] = actual
    prediction_df["Predicted"] = predictions
    prediction_df.to_csv(
        OUTPUT_DIR / f"{model_type}_predictions.csv",
        index=False,
    )

    pd.DataFrame(history.history).to_csv(
        OUTPUT_DIR / f"{model_type}_training_history.csv",
        index=False,
    )

    return metrics


# 4. Save model files


def save_predictions(scats_numbers, datetimes, actual, predicted, filename: str):
    """Save model predictions so they can be checked in Excel or VS Code."""

    prediction_df = pd.DataFrame(
        {
            "SCATS Number": scats_numbers,
            "Datetime": datetimes,
            "Actual": actual,
            "Predicted": predicted,
        }
    )

    prediction_df.to_csv(OUTPUT_DIR / filename, index=False)


def save_model_comparison(results: list[dict]):
    """Save the final comparison table for the report."""

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df[
        ["Model", "MAE", "RMSE", "MAPE_percent", "R2", "TrainingSeconds"]
    ]

    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "model_comparison.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(comparison_df)


# 5. Main training function


def main(models: list[str], epochs: int):
    """Run the selected models."""

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_time_series(TIME_SERIES_DATA)
    results = []

    if "xgb" in models:
        results.append(train_xgboost(df))

    if "lstm" in models:
        results.append(train_deep_model(df, "lstm", epochs))

    if "gru" in models:
        results.append(train_deep_model(df, "gru", epochs))

    save_model_comparison(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train traffic prediction models for Assignment 2B"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "lstm", "gru"],
        choices=["xgb", "lstm", "gru"],
        help="Choose which models to train: xgb, lstm, gru",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for LSTM and GRU",
    )

    args = parser.parse_args()
    main(args.models, args.epochs)
