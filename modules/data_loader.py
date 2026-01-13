import pandas as pd
import json
import numpy as np
import os

def load_from_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def load_from_json(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return df

def extract_user_data(data_source, mode="json"):
    if mode == "csv":
        df = load_from_csv(data_source)
        return df
    elif mode == "json":
        df = load_from_json(data_source)
        return df
    else:
        raise ValueError("Unsupported data source type")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("NaN", np.nan)
    df = df.fillna(df.mean(numeric_only=True))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

def get_user_record(df: pd.DataFrame, participant_id: str) -> dict:
    record = df[df["participant_id"] == participant_id]
    if record.empty:
        raise ValueError(f"Participant {participant_id} not found.")
    return record.to_dict(orient="records")[0]
