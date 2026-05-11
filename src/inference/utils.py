import json
from pathlib import Path

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

LABEL_MAP = {
    0: "OK",
    1: "MEDIUM",
    2: "HIGH",
}


def save_metadata(path="models/inference_metadata.json"):
    meta = {
        "feature_cols": FEATURE_COLS,
        "label_map": LABEL_MAP,
        "input_dim": len(FEATURE_COLS),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)