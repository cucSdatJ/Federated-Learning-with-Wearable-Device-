import numpy as np

def map_activity(activity_id: int) -> str:
    if activity_id in [1, 2, 3, 17]:
        return "rest"
    elif activity_id in [4, 16]:
        return "walk"
    elif activity_id in [6, 7, 12, 13]:
        return "brisk"
    elif activity_id in [5, 24]:
        return "run"
    else:
        return "ignore"

def label_3class(hr: float, activity_name: str) -> int:
    med = {"rest": 90, "walk": 105, "brisk": 120, "run": 150}
    high = {"rest": 100, "walk": 120, "brisk": 130, "run": 170}

    if activity_name not in med:
        raise ValueError(f"Unknown activity_name: {activity_name}")

    if hr >= high[activity_name]:
        return 2
    elif hr >= med[activity_name]:
        return 1
    return 0

def activity_onehot(activity_name: str) -> dict:
    return {
        "act_rest": 1 if activity_name == "rest" else 0,
        "act_walk": 1 if activity_name == "walk" else 0,
        "act_brisk": 1 if activity_name == "brisk" else 0,
        "act_run": 1 if activity_name == "run" else 0,
    }

def acc_magnitude(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)