from pathlib import Path
import pandas as pd
import numpy as np

from src.data.utils import map_activity, label_3class, activity_onehot, acc_magnitude

COLS = {
    0: "timestamp",
    1: "activity_id",
    2: "heart_rate",
    4: "acc_x",
    5: "acc_y",
    6: "acc_z",
}

def load_subject(file_path: str, subject_id: int) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")

    df = df[list(COLS.keys())].rename(columns=COLS)

    # convert numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # discard transient activity
    df = df[df["activity_id"] != 0].copy()

    # interpolate HR
    df["heart_rate"] = df["heart_rate"].interpolate(method="linear")
    df["heart_rate"] = df["heart_rate"].bfill().ffill()

    # remove rows still missing essential values
    df = df.dropna(subset=["heart_rate", "acc_x", "acc_y", "acc_z"])

    # map activity
    df["activity_name"] = df["activity_id"].apply(map_activity)

    # features
    df["hr_rolling_mean"] = df["heart_rate"].rolling(window=10, min_periods=1).mean()
    df["hr_rolling_std"] = df["heart_rate"].rolling(window=10, min_periods=1).std().fillna(0.0)

    df["acc_magnitude"] = acc_magnitude(df["acc_x"], df["acc_y"], df["acc_z"])

    onehot = df["activity_name"].apply(activity_onehot).apply(pd.Series)
    df = pd.concat([df, onehot], axis=1)

    # label
    df["label"] = df.apply(
        lambda row: label_3class(row["heart_rate"], row["activity_name"]),
        axis=1
    )

    df["subject_id"] = subject_id

    final_cols = [
        "subject_id",
        "timestamp",
        "heart_rate",
        "hr_rolling_mean",
        "hr_rolling_std",
        "acc_magnitude",
        "act_rest",
        "act_walk",
        "act_brisk",
        "act_run",
        "label",
    ]

    return df[final_cols].reset_index(drop=True)