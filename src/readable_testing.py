from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)

testing_text = """
Readable Model Testing File for Assignment 2B

This file explains the testing and evaluation of the machine learning models used for traffic flow prediction.

Models tested:
1. XGBoost
2. LSTM
3. GRU

The actual trained model files are saved as .joblib and .keras files.
These files are not readable like normal text files because they store the trained model in machine-readable format.

Testing approach:
The models were trained using the SCATS traffic dataset and tested by comparing their predicted traffic flow against the actual traffic flow values.

Evaluation metrics used:
- MAE: shows average prediction error in the same units as traffic flow, giving equal weight to all errors.
- RMSE: will have predication errors as well as larger errors more heavily, showing how well the model handles large errors.
- MAPE: the prediction error to actual value ratio, expressed as a percentage, showing the average percentage error of predictions.
- R2: demonstrates how well the model explains the variance in the data, with 1 being perfect and 0 meaning no better than predicting the mean.
Purpose of testing:
the reasoning behind testing models is to evaluate how well they can predict traffic flow and to compare their performance using the evaluation metrics.

To run model training and testing again:
python3 src/main.py train --models xgb lstm gru --epochs 10
"""

with open(MODEL_DIR / "model_testing.txt", "w", encoding="utf-8") as file:
    file.write(testing_text)

xgboost_testing = {
    "model": "XGBoost",
    "testing_purpose": "Test how well XGBoost predicts SCATS traffic flow using the trained model.",
    "saved_model_file": "xgboost.joblib",
    "evaluation_metrics": ["MAE", "RMSE", "MAPE_percent", "R2", "TrainingSeconds"],
    "parameters": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror"
    }
}

with open(MODEL_DIR / "xgboost_testing.json", "w", encoding="utf-8") as file:
    json.dump(xgboost_testing, file, indent=4)

lstm_testing = """
LSTM Model Testing

Model type:
The kind of model is a deep learning time-series model.

Testing purpose:
The LSTM model was tested to check how well it can learn traffic patterns over time and predict future traffic flow.

Saved files:
- lstm_model.keras
- lstm_scalers.joblib

Evaluation metrics:
- MAE
- RMSE
- MAPE_percent
- R2
- TrainingSeconds

The LSTM model is used for traffic flow prediction because it can capture long-term dependencies in sequential data, making it suitable for time-series forecasting tasks like traffic flow prediction.
"""

with open(MODEL_DIR / "lstm_testing.txt", "w", encoding="utf-8") as file:
    file.write(lstm_testing)

gru_testing = """
GRU Model Testing

Model type:
The kind of model is a deep learning time-series model.

Testing purpose:
The GRU model was tested to check how well it can predict traffic flow using previous traffic patterns.

Saved files:
- gru_model.keras
- gru_scalers.joblib

Evaluation metrics:
- MAE
- RMSE
- MAPE_percent
- R2
- TrainingSeconds

The GRU model is similar to LSTM, but it has a simpler structure and trains somewhat faster. It is useful for traffic flow prediction because it can capture temporal dependencies in the data while being more efficient than LSTM.
"""

with open(MODEL_DIR / "gru_testing.txt", "w", encoding="utf-8") as file:
    file.write(gru_testing)

comparison_file = OUTPUT_DIR / "model_comparison.csv"

if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)
    comparison.to_csv(MODEL_DIR / "readable_model_testing.csv", index=False)
    comparison.to_json(MODEL_DIR / "readable_model_testing.json", orient="records", indent=4)
    print("Created model testing files and readable model testing results.")
else:
    print("Created model testing text files.")
    print("model_comparison.csv was not found yet. Train the models first.")