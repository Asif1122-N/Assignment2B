"""The training module for machine learning models."""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE, TIME_SERIES_DATA, LOOKBACK_STEPS
from data_processing import load_time_series, make_sequence_data, make_tabular_features, temporal_train_test_split


def regression_metrics(y_true, y_pred):
    """Calculate model accuracy scores for the report comparison table."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    # MAPE shows the average percentage error.
    # The non-zero check avoids division by zero errors.
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
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


def train_random_forest(df: pd.DataFrame) -> dict:
    """Train the Random Forest model."""
    data, feature_cols = make_tabular_features(df)
    train_df, test_df = temporal_train_test_split(data, TEST_SIZE)

    X_train, y_train = train_df[feature_cols], train_df["Traffic"]
    X_test, y_test = test_df[feature_cols], test_df["Traffic"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    metrics.update({"Model": "Random Forest", "TrainingSeconds": train_seconds})

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        MODEL_DIR / "random_forest.joblib",
    )

    pd.DataFrame(
        {
            "SCATS Number": test_df["SCATS Number"].values,
            "Datetime": test_df["Datetime"].values,
            "Actual": y_test.values,
            "Predicted": pred,
        }
    ).to_csv(OUTPUT_DIR / "random_forest_predictions.csv", index=False)

    return metrics


def train_xgboost(df: pd.DataFrame) -> dict:
    """Train the XGBoost model.

    XGBoost is used as the third ML model so we can compare it with LSTM and GRU.
    The import is placed inside this function so the other models can still run
    even if XGBoost is not installed correctly.
    """
    from xgboost import XGBRegressor

    data, feature_cols = make_tabular_features(df)
    train_df, test_df = temporal_train_test_split(data, TEST_SIZE)

    X_train, y_train = train_df[feature_cols], train_df["Traffic"]
    X_test, y_test = test_df[feature_cols], test_df["Traffic"]

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

    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    metrics.update({"Model": "XGBoost", "TrainingSeconds": train_seconds})

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        MODEL_DIR / "xgboost.joblib",
    )

    pd.DataFrame(
        {
            "SCATS Number": test_df["SCATS Number"].values,
            "Datetime": test_df["Datetime"].values,
            "Actual": y_test.values,
            "Predicted": pred,
        }
    ).to_csv(OUTPUT_DIR / "xgboost_predictions.csv", index=False)

    return metrics


def build_deep_model(model_type: str, input_shape: tuple[int, int]):
    """Build either an LSTM or GRU deep learning model."""
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, Input
    from tensorflow.keras.models import Sequential

    recurrent_layer = LSTM if model_type.lower() == "lstm" else GRU

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
    callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    return model, callback


def train_deep_model(
    df: pd.DataFrame,
    model_type: str,
    epochs: int = 30,
    batch_size: int = 64,
) -> dict:
    """Train either the LSTM or GRU model."""
    seq = make_sequence_data(df, lookback=LOOKBACK_STEPS, test_size=TEST_SIZE)
    model, callback = build_deep_model(model_type, input_shape=seq.X_train.shape[1:])

    start = time.time()
    history = model.fit(
        seq.X_train,
        seq.y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=1,
    )
    train_seconds = time.time() - start

    pred_scaled = model.predict(seq.X_test).reshape(-1, 1)
    pred = seq.target_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual = seq.target_scaler.inverse_transform(seq.y_test).reshape(-1)

    metrics = regression_metrics(actual, pred)
    metrics.update(
        {
            "Model": model_type.upper(),
            "TrainingSeconds": train_seconds,
        }
    )

    model.save(MODEL_DIR / f"{model_type.lower()}_model.keras")

    joblib.dump(
        {
            "feature_scaler": seq.feature_scaler,
            "target_scaler": seq.target_scaler,
            "feature_cols": seq.feature_cols,
            "lookback": LOOKBACK_STEPS,
        },
        MODEL_DIR / f"{model_type.lower()}_scalers.joblib",
    )

    pred_df = seq.test_meta.copy()
    pred_df["Actual"] = actual
    pred_df["Predicted"] = pred
    pred_df.to_csv(OUTPUT_DIR / f"{model_type.lower()}_predictions.csv", index=False)

    pd.DataFrame(history.history).to_csv(
        OUTPUT_DIR / f"{model_type.lower()}_training_history.csv",
        index=False,
    )

    return metrics


def main(models: list[str], epochs: int):
    """Train the selected models and save the comparison results."""
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_time_series(TIME_SERIES_DATA)
    all_metrics = []

    if "rf" in models:
        all_metrics.append(train_random_forest(df))

    if "xgb" in models:
        all_metrics.append(train_xgboost(df))

    if "lstm" in models:
        all_metrics.append(train_deep_model(df, "lstm", epochs=epochs))

    if "gru" in models:
        all_metrics.append(train_deep_model(df, "gru", epochs=epochs))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[
        ["Model", "MAE", "RMSE", "MAPE_percent", "R2", "TrainingSeconds"]
    ]

    metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(metrics_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "lstm", "gru"],
        choices=["rf", "xgb", "lstm", "gru"],
        help="Choose which models to train: rf, xgb, lstm, gru",
    )
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    main(args.models, args.epochs)
