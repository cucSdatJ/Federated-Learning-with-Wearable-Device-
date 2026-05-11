import math
from typing import Dict, List, Optional

import numpy as np


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


# PAMAP2 activity grouping used in this project
# Based on the current project design:
# - rest  = lying, sitting, standing, ironing
# - walk  = walking, vacuum_cleaning
# - brisk = cycling, nordic_walking, ascending/descending stairs
# - run   = running, rope_jumping
ACTIVITY_ID_TO_GROUP = {
    1: "rest",   # lying
    2: "rest",   # sitting
    3: "rest",   # standing
    17: "rest",  # ironing

    4: "walk",   # walking
    16: "walk",  # vacuum_cleaning

    6: "brisk",  # cycling
    7: "brisk",  # nordic_walking
    12: "brisk", # ascending_stairs
    13: "brisk", # descending_stairs

    5: "run",    # running
    24: "run",   # rope_jumping
}

# Device activity code from CE firmware
# currentActivity:
# 0 = Resting / slight movement
# 1 = Light Walk
# 2 = Brisk Walk
# 4 = Running / vigorous shaking
DEVICE_ACTIVITY_CODE_TO_GROUP = {
    0: "rest",
    1: "walk",
    2: "brisk",
    4: "run",
}

def compute_acc_magnitude(acc_x: float, acc_y: float, acc_z: float) -> float:
    """Compute accelerometer magnitude."""
    return float(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))


def compute_hr_stats(hr_window: List[float]) -> tuple[float, float]:
    """
    Compute rolling heart-rate mean/std from a recent window.
    If the window is empty, fallback to (0.0, 0.0).
    """
    if hr_window is None or len(hr_window) == 0:
        return 0.0, 0.0

    arr = np.asarray(hr_window, dtype=np.float32)
    return float(np.mean(arr)), float(np.std(arr))


def activity_group_to_onehot(activity_group: str) -> Dict[str, int]:
    """
    Convert one of: rest/walk/brisk/run into one-hot encoding.
    """
    valid = {"rest", "walk", "brisk", "run"}
    if activity_group not in valid:
        raise ValueError(f"Invalid activity_group='{activity_group}'. Must be one of {valid}")

    return {
        "act_rest": 1 if activity_group == "rest" else 0,
        "act_walk": 1 if activity_group == "walk" else 0,
        "act_brisk": 1 if activity_group == "brisk" else 0,
        "act_run": 1 if activity_group == "run" else 0,
    }


def map_activity_id_to_group(activity_id: int) -> str:
    """
    Map PAMAP2 activity_id to one of the 4 deployment groups.
    """
    if activity_id not in ACTIVITY_ID_TO_GROUP:
        raise ValueError(
            f"Unsupported activity_id={activity_id}. "
            f"Supported IDs: {sorted(ACTIVITY_ID_TO_GROUP.keys())}"
        )
    return ACTIVITY_ID_TO_GROUP[activity_id]

def map_device_activity_code_to_group(activity_code: int) -> str:
    """
    Map CE firmware currentActivity code to one of:
    rest / walk / brisk / run
    """
    if activity_code not in DEVICE_ACTIVITY_CODE_TO_GROUP:
        raise ValueError(
            f"Unsupported device activity code={activity_code}. "
            f"Supported codes: {sorted(DEVICE_ACTIVITY_CODE_TO_GROUP.keys())}"
        )
    return DEVICE_ACTIVITY_CODE_TO_GROUP[activity_code]

def build_feature_dict(
    heart_rate: float,
    hr_window: List[float],
    acc_x: float,
    acc_y: float,
    acc_z: float,
    activity_group: str,
    hour_of_day: Optional[float] = None,
    timestamp_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """
    Build the 10-feature input dictionary expected by the model.

    You must provide either:
    - hour_of_day
    OR
    - timestamp_seconds
    """

    hr_mean, hr_std = compute_hr_stats(hr_window)
    acc_mag = compute_acc_magnitude(acc_x, acc_y, acc_z)
    onehot = activity_group_to_onehot(activity_group)

    feature_dict = {
        "heart_rate": float(heart_rate),
        "hr_rolling_mean": float(hr_mean),
        "hr_rolling_std": float(hr_std),
        "acc_magnitude": float(acc_mag),
        **onehot,
    }

    return feature_dict


def build_feature_dict_from_activity_id(
    heart_rate: float,
    hr_window: List[float],
    acc_x: float,
    acc_y: float,
    acc_z: float,
    activity_id: int,
    hour_of_day: Optional[float] = None,
    timestamp_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """
    Same as build_feature_dict, but accepts PAMAP2 activity_id directly.
    """
    activity_group = map_activity_id_to_group(activity_id)
    return build_feature_dict(
        heart_rate=heart_rate,
        hr_window=hr_window,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        activity_group=activity_group,
        hour_of_day=hour_of_day,
        timestamp_seconds=timestamp_seconds,
    )

def build_feature_dict_from_device_code(
    heart_rate: float,
    hr_window: List[float],
    acc_x: float,
    acc_y: float,
    acc_z: float,
    activity_code: int,
    hour_of_day: Optional[float] = None,
    timestamp_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """
    Build feature dict directly from CE firmware currentActivity code.
    """
    activity_group = map_device_activity_code_to_group(activity_code)
    return build_feature_dict(
        heart_rate=heart_rate,
        hr_window=hr_window,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        activity_group=activity_group,
        hour_of_day=hour_of_day,
        timestamp_seconds=timestamp_seconds,
    )


def feature_dict_to_ordered_vector(feature_dict: Dict[str, float]) -> List[float]:
    """
    Convert feature dict into ordered list matching FEATURE_COLS.
    """
    missing = [col for col in FEATURE_COLS if col not in feature_dict]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return [float(feature_dict[col]) for col in FEATURE_COLS]


if __name__ == "__main__":
    # Example usage
    sample = build_feature_dict(
        heart_rate=128.0,
        hr_window=[122.0, 124.0, 126.0, 128.0, 130.0],
        acc_x=0.8,
        acc_y=9.5,
        acc_z=1.7,
        activity_group="brisk",
        hour_of_day=14.5,
    )

    print("Feature dict:")
    for k, v in sample.items():
        print(f"  {k}: {v}")

    print("\nOrdered vector:")
    print(feature_dict_to_ordered_vector(sample))