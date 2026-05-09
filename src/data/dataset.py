from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLS = [
    "heart_rate",
    "hr_rolling_mean",
    "hr_rolling_std",
    "acc_magnitude",
    "hour_sin",
    "hour_cos",
    "act_rest",
    "act_walk",
    "act_brisk",
    "act_run",
]

TARGET_COL = "label"


class WearableDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.y = torch.tensor(df[TARGET_COL].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)