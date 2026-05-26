"""Configuration values for Assignment 2B TBRGS."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLEANED_DATA = DATA_DIR / "cleaned_scats_data.csv"
TIME_SERIES_DATA = DATA_DIR / "scats_time_series.csv"

LOOKBACK_STEPS = 8
TEST_SIZE = 0.2
RANDOM_STATE = 42

SPEED_LIMIT_KMH = 60.0
INTERSECTION_DELAY_SECONDS = 30.0
