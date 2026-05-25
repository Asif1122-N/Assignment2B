import joblib
import pandas as pd
import math

# Load saved XGBoost object
saved_data = joblib.load("models/xgboost.joblib")

# Extract model from dictionary
model = saved_data["model"]

# Example: Sunday 9:00 AM at SCATS 3001
hour = 9
minute = 0
day_of_week = 6   # Monday = 0, Sunday = 6

# Convert time into cyclical values
minutes_in_day = hour * 60 + minute
time_sin = math.sin(2 * math.pi * minutes_in_day / 1440)
time_cos = math.cos(2 * math.pi * minutes_in_day / 1440)

# Convert day into cyclical values
day_sin = math.sin(2 * math.pi * day_of_week / 7)
day_cos = math.cos(2 * math.pi * day_of_week / 7)

# Input must match EXACT training columns
test_input = pd.DataFrame([{
    "SCATS Number": 3001,
    "Hour": hour,
    "Minute": minute,
    "DayOfWeek": day_of_week,
    "IsWeekend": 1,

    "TimeSin": time_sin,
    "TimeCos": time_cos,
    "DaySin": day_sin,
    "DayCos": day_cos,

    # Previous traffic values
    "Lag1": 150,
    "Lag2": 145,
    "Lag4": 140,
    "Lag8": 135,
    "Lag96": 120,

    # Rolling average traffic
    "RollingMean4": 142,
    "RollingMean8": 138
}])

# Predict
prediction = model.predict(test_input)

print("Predicted traffic flow:", round(prediction[0]))
print("Meaning: around", round(prediction[0]), "vehicles are predicted.")