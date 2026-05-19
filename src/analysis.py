"""Create comparison plots/tables for the report after training models."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR


def plot_model_comparison():
    metrics_path = OUTPUT_DIR / "model_comparison.csv"
    metrics = pd.read_csv(metrics_path)

    for metric in ["MAE", "RMSE", "MAPE_percent", "R2", "TrainingSeconds"]:
        plt.figure(figsize=(7, 4))
        plt.bar(metrics["Model"], metrics[metric])
        plt.title(f"Model comparison by {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"comparison_{metric}.png", dpi=200)
        plt.close()


def plot_actual_vs_predicted(model_file_name: str, title: str, rows: int = 500):
    df = pd.read_csv(OUTPUT_DIR / model_file_name).head(rows)
    plt.figure(figsize=(10, 4))
    plt.plot(df["Actual"].values, label="Actual")
    plt.plot(df["Predicted"].values, label="Predicted")
    plt.title(title)
    plt.xlabel("Test sample")
    plt.ylabel("Traffic flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{Path(model_file_name).stem}_actual_vs_predicted.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_model_comparison()
    for file_name, title in [
        ("random_forest_predictions.csv", "Random Forest actual vs predicted"),
        ("lstm_predictions.csv", "LSTM actual vs predicted"),
        ("gru_predictions.csv", "GRU actual vs predicted"),
    ]:
        path = OUTPUT_DIR / file_name
        if path.exists():
            plot_actual_vs_predicted(file_name, title)
    print(f"Analysis charts saved in {OUTPUT_DIR}")
