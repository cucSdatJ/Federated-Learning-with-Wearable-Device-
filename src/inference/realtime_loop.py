import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import collections
import joblib
import numpy as np
import torch
import serial
import pandas as pd
from src.inference.logger import init_db, log_reading, log_alert
from src.models.mlp import MLPClassifier
from src.inference.utils import FEATURE_COLS, LABEL_MAP

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH  = "models/flower_global_best.pt"
SCALER_PATH = "models/flower_scaler.pkl"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HR_WINDOW_SIZE = 10

# Map device activity code → activity group (khớp firmware CE)
ACTIVITY_MAP = {
    0: "rest",
    1: "walk",
    2: "brisk",
    4: "run",
}

# One-hot encoding theo thứ tự FEATURE_COLS
ONEHOT_MAP = {
    "rest":  [1, 0, 0, 0],
    "walk":  [0, 1, 0, 0],
    "brisk": [0, 0, 1, 0],
    "run":   [0, 0, 0, 1],
}


# ==========================================
# LOAD MODEL
# ==========================================
def load_model(model_path, scaler_path):
    model = MLPClassifier(input_dim=len(FEATURE_COLS), num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    scaler = joblib.load(scaler_path)
    print(f"[*] Model loaded: {model_path}")
    print(f"[*] Scaler loaded: {scaler_path}")
    return model, scaler


# ==========================================
# BUILD FEATURES
# ==========================================
def build_features(bpm: float, activity_code: int, hr_window: collections.deque) -> np.ndarray:
    # Rolling stats từ window (giống fix Flutter)
    window_list = list(hr_window)
    hr_mean = float(np.mean(window_list))
    hr_std  = float(np.std(window_list))

    # Acc magnitude — không có sensor thật, dùng giá trị trung bình theo activity
    acc_defaults = {"rest": 9.8, "walk": 10.5, "brisk": 12.0, "run": 14.0}
    activity_group = ACTIVITY_MAP.get(activity_code, "rest")
    acc_magnitude = acc_defaults[activity_group]

    # One-hot
    onehot = ONEHOT_MAP[activity_group]

    # Ghép thành vector 8 features theo đúng thứ tự FEATURE_COLS:
    # heart_rate, hr_rolling_mean, hr_rolling_std, acc_magnitude,
    # act_rest, act_walk, act_brisk, act_run
    features = np.array([
        bpm,
        hr_mean,
        hr_std,
        acc_magnitude,
        *onehot,
    ], dtype=np.float32)

    return features


# ==========================================
# PREDICT
# ==========================================
def predict(model, scaler, features: np.ndarray):
    x = scaler.transform(pd.DataFrame(features.reshape(1,-1), columns=FEATURE_COLS))
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    return pred_class, probs


# ==========================================
# SERIAL LOOP
# ==========================================
def run_serial(port: str, baud: int = 115200):

    model, scaler = load_model(MODEL_PATH, SCALER_PATH)
    conn = init_db()
    # Rolling HR window — khởi tạo với giá trị resting mặc định
    hr_window = collections.deque(maxlen=HR_WINDOW_SIZE)
    for _ in range(HR_WINDOW_SIZE):
        hr_window.append(72.0)

    print(f"[*] Connecting to {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2)  # Chờ ESP32 boot
    print(f"[*] Connected! Listening for BPM,Activity...\n")

    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()

            # Bỏ qua dòng không đúng format
            if "," not in raw:
                continue
            parts = raw.split(",")
            if len(parts) != 2:
                continue

            try:
                bpm           = float(parts[0])
                activity_code = int(parts[1])
            except ValueError:
                continue

            # Sanity check
            if not (30 <= bpm <= 220):
                continue
            if activity_code not in ACTIVITY_MAP:
                continue

            # Cập nhật rolling window
            hr_window.append(bpm)

            # Build features + predict
            features   = build_features(bpm, activity_code, hr_window)
            pred_class, probs = predict(model, scaler, features)
            pred_label = LABEL_MAP[pred_class]

            # Gửi lệnh về ESP32
            if pred_class == 2:
                ser.write(b'A')   # HIGH
                cmd = 'A'
            elif pred_class == 1:
                ser.write(b'M')   # MEDIUM
                cmd = 'M'
            else:
                ser.write(b'O')   # OK
                cmd = 'O'

            # Log ra terminal
            ts = time.strftime("%H:%M:%S")
            print(
                f"[{ts}] BPM={bpm:6.1f} | "
                f"Activity={ACTIVITY_MAP.get(activity_code, '?'):5s} | "
                f"→ {pred_label:6s} "
                f"(OK={probs[0]:.2f} MED={probs[1]:.2f} HIGH={probs[2]:.2f}) "
                f"| CMD={cmd}"
            )
            log_reading(conn, bpm, activity_code, probs.tolist(), pred_label)
            if pred_class in [1, 2]:
                log_alert(conn, bpm, severity=pred_label)

    except KeyboardInterrupt:
        print("\n[*] Stopped by user.")
    finally:
        ser.write(b'O')  # Reset ESP32 về safe state
        ser.close()
        conn.close()
        print("[*] Serial closed.")


# ==========================================
# DEMO MODE (không cần ESP32)
# ==========================================
def run_demo():
    model, scaler = load_model(MODEL_PATH, SCALER_PATH)

    test_cases = [
        (75,  0, "Resting normal"),
        (95,  0, "Resting elevated"),
        (102, 0, "Resting HIGH"),
        (108, 1, "Walking normal"),
        (122, 1, "Walking elevated"),
        (125, 2, "Brisk normal"),
        (138, 2, "Brisk elevated"),
        (155, 4, "Running normal"),
        (165, 4, "Running elevated"),
        (175, 4, "Running HIGH"),
    ]

    print(f"\n{'='*65}")
    print(f"{'BPM':>6} {'Activity':>8} {'Label':>8} {'OK':>6} {'MED':>6} {'HIGH':>6}  Description")
    print(f"{'='*65}")

    for bpm, act_code, desc in test_cases:
        # Build window realistic
        hr_window = collections.deque(
            [bpm + x for x in [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]],
            maxlen=HR_WINDOW_SIZE
        )
        features = build_features(float(bpm), act_code, hr_window)
        pred_class, probs = predict(model, scaler, features)
        label = LABEL_MAP[pred_class]

        print(
            f"{bpm:>6} {ACTIVITY_MAP[act_code]:>8} {label:>8} "
            f"{probs[0]:>6.3f} {probs[1]:>6.3f} {probs[2]:>6.3f}  {desc}"
        )

    print(f"{'='*65}\n")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=str, default=None,
                        help="Serial port (e.g. COM5). Bỏ trống để chạy demo mode.")
    parser.add_argument("--baud",  type=int, default=115200)
    args = parser.parse_args()

    if args.port is None:
        print("[*] No port specified → running demo mode")
        run_demo()
    else:
        run_serial(args.port, args.baud)