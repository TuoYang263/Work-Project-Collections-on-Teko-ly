import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "output"

def load_route_window():
    return pd.read_parquet(DATA_DIR / "gold_route_window.parquet")

def load_route_daily():
    return pd.read_parquet(DATA_DIR / "gold_route_daily.parquet")

def load_pipeline_metrics():
    return pd.read_parquet(DATA_DIR / "pipeline_metrics.parquet")