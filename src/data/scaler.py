import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "heart_rate",
    "hr_rolling_mean",
    "hr_rolling_std",
    "acc_magnitude",
    "act_rest",
    "act_walk",
    "act_brisk",
    "act_run",
]


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    return scaler


def transform_df(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[FEATURE_COLS] = scaler.transform(out[FEATURE_COLS])
    return out