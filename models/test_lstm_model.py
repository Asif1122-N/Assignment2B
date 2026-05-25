import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# =========================
# 1. Load model and scaler
# =========================

model = load_model("models/lstm_model.keras")
scalers = joblib.load("models/lstm_scalers.joblib")

feature_scaler = scalers["feature_scaler"]
target_scaler = scalers["target_scaler"]
feature_cols = scalers["feature_cols"]
lookback = scalers["lookback"]

print("Expected feature columns:", feature_cols)
print("Lookback:", lookback)

# =========================
# 2. Load dataset
# =========================

df = pd.read_csv("data/cleaned_scats_data.csv")

print("\nOriginal columns:")
print(df.columns.tolist())

# =========================
# 3. Rename traffic column if needed
# =========================

possible_traffic_cols = [
    "Traffic",
    "traffic",
    "TrafficVolume",
    "traffic_volume",
    "Volume",
    "volume",
    "VOLUME",
]

traffic_col = None

for col in possible_traffic_cols:
    if col in df.columns:
        traffic_col = col
        break

if traffic_col is None:
    raise ValueError(
        "Could not find traffic column. Check your CSV column names above."
    )

df = df.rename(columns={traffic_col: "Traffic"})


# 4. Create time features


possible_time_cols = [
    "DateTime",
    "Datetime",
    "datetime",
    "Date",
    "date",
    "Time",
    "time",
]

time_col = None

for col in possible_time_cols:
    if col in df.columns:
        time_col = col
        break

if time_col is not None:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df["Hour"] = df[time_col].dt.hour
    df["DayOfWeek"] = df[time_col].dt.dayofweek
else:
    if "Hour" not in df.columns:
        raise ValueError(
            "No Date/Time column or Hour column found. Cannot create TimeSin/TimeCos."
        )

    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = 0

# Cyclical time encoding
df["TimeSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["TimeCos"] = np.cos(2 * np.pi * df["Hour"] / 24)

df["DaySin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
df["DayCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

# =========================
# 5. Check required features
# =========================

missing_cols = [col for col in feature_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df.dropna(subset=feature_cols)

# =========================
# 6. Prepare LSTM input
# =========================

data = df[feature_cols].values

data_scaled = feature_scaler.transform(data)

X = []
y = []

traffic_index = feature_cols.index("Traffic")

for i in range(lookback, len(data_scaled)):
    X.append(data_scaled[i - lookback:i])
    y.append(data_scaled[i, traffic_index])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 7. Predict
# =========================

y_pred_scaled = model.predict(X)

y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual = target_scaler.inverse_transform(y)

# =========================
# 8. Evaluate
# =========================

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)
mape = mean_absolute_percentage_error(y_actual, y_pred)

approx_accuracy = 100 - (mape * 100)

print("\n===== LSTM Regression Evaluation =====")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
print("MAPE:", round(mape * 100, 2), "%")
print("Approx Accuracy:", round(approx_accuracy, 2), "%")