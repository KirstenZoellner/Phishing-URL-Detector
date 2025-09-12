from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = {"url", "label"}

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    if missing:
        raise ValueError(f"CSV must contain columns {REQUIRED_COLUMNS}, but missing: {missing}")
    # Basic clean
    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    # Coerce label to int 0/1
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label column must contain only 0 (legit) or 1 (phishing)")
    # Drop duplicates
    df = df.drop_duplicates(subset=["url"])
    return df