# FL Wearable

Federated Learning for Personalized Wearable Device Control using **PAMAP2**, **PyTorch**, **Flower**, **FastAPI**, **ESP32**, and **Flutter**.

## Overview

This project builds a privacy-preserving wearable health monitoring pipeline where multiple clients collaboratively train a global model **without sharing raw physiological data**. The system classifies wearable sensor readings into three alert levels:

- `OK`
- `MEDIUM`
- `HIGH`

The project started from a Python + ESP32 serial inference flow, but the **current end-to-end architecture has been updated to Option B**, where the **Flutter mobile app acts as the main real-time bridge** between ESP32 hardware and the backend inference API.

### Current end-to-end architecture

```text
ESP32 (Sensors)
  └── WiFi → MQTT publish → broker.hivemq.com:1883
                                    │
                         topic: wearable/data
                                    │
                              Flutter App
                           (SensorProvider)
                                    │
                      POST /predict-from-sensor
                                    │
                      FastAPI Backend (Laptop/PC)
                         flower_global_best.pt
                                    │
                    response: OK / MEDIUM / HIGH
                                    │
                              Flutter App
                        (PredictionProvider)
                                    │
                      MQTT publish → wearable/alert
                                    │
                                 ESP32
                  LED + Buzzer react to alert severity
```


---

## Main Contributions

- PAMAP2-based data preprocessing and activity-aware pseudo-labeling[38]
- 3-class wearable alert classification (`OK`, `MEDIUM`, `HIGH`)
- Centralized baseline training[38]
- Local-only baseline training[38]
- Federated Learning with Flower[38]
- FastAPI inference backend for real-time prediction
- Flutter mobile app for live monitoring, prediction, history, and notifications
- ESP32 wearable integration with MQTT-based real-time communication

---

## Project Status

### Completed
- ESP32 firmware (basic version)
- PAMAP2 data pipeline
- MLP model training pipeline
- Centralized training
- Local-only training
- Federated Learning with Flower
- Evaluation and plots
- FastAPI inference backend
- SQLite logger
- Flutter app structure and main UI screens

### In progress
- ESP32 firmware adaptation for Option B MQTT flow
- Flutter ↔ ESP32 integration via MQTT
- Final report writing

### Optional / not core
- Adafruit IO dashboard is not part of the main architecture and has not been started

---

## Dataset

We use the **PAMAP2 Physical Activity Monitoring** dataset.

Download:
https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

Place raw files under:

```text
data/raw/
├── subject101.dat
├── subject102.dat
├── ...
└── subject109.dat
```

### Activity grouping

Activities are grouped into 4 intensity levels based on observed mean heart rate:

| Group | Activity IDs | Mean HR range |
|---|---|---|
| `rest` | 1, 2, 3, 17 | 75–90 bpm |
| `walk` | 4, 16 | 104–113 bpm |
| `brisk` | 6, 7, 12, 13 | 124–130 bpm |
| `run` | 5, 24 | 157–162 bpm |

### Labeling strategy

Since PAMAP2 does not provide medical alert labels, this project generates **pseudo-labels** using activity-aware heart-rate thresholds:

| Group | MEDIUM (≥) | HIGH (≥) |
|---|---:|---:|
| rest | 90 | 100 |
| walk | 105 | 120 |
| brisk | 120 | 135 |
| run | 150 | 170 |

Label mapping:
- `0 = OK`
- `1 = MEDIUM`
- `2 = HIGH`

> Note: these are **engineered pseudo-labels**, not clinically validated risk labels.

---

## Features Used by the Model

The **current codebase uses 8 input features**.

### Final feature set

```text
heart_rate
hr_rolling_mean
hr_rolling_std
acc_magnitude
act_rest
act_walk
act_brisk
act_run
```

This 8-feature configuration is the one actually used in the current implementation.

---

## Model Architecture

The **actual current MLP** is:

```text
MLPClassifier(input_dim=8, num_classes=3):
  Linear(8, 32) → ReLU → Dropout(0.2)
  Linear(32, 16) → ReLU → Dropout(0.1)
  Linear(16, 3)
```

---

## Repository Structure

```text
fl_wearable/
├── data/
│   ├── raw/
│   └── processed/
├── database/
│   └── schema.sql
├── experiments/
│   ├── compare_summary.csv
│   ├── fl_round_metrics.csv
│   └── plots/
├── hardware/
│   └── src/
│       ├── main.cpp
│       ├── pulse_sensor.cpp
│       ├── motion_sensor.cpp
│       ├── comm_task.cpp
│       ├── alert_manager.cpp
│       ├── config.h
│       └── globals.h / globals.cpp
├── models/
│   ├── flower_global_best.pt
│   ├── flower_scaler.pkl
│   ├── centralized_mlp.pt
│   └── local_client_*.pt
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── pamap2_loader.py
│   │   ├── preprocess.py
│   │   ├── scaler.py
│   │   ├── split_clients.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── compare_results.py
│   │   └── plot_results.py
│   ├── fl/
│   │   ├── client.py
│   │   ├── common.py
│   │   └── server.py
│   ├── inference/
│   │   ├── api_server.py
│   │   ├── export_model.py
│   │   ├── feature_builder.py
│   │   ├── logger.py
│   │   ├── predict.py
│   │   ├── realtime_loop.py
│   │   └── utils.py
│   ├── models/
│   │   └── mlp.py
│   └── training/
│       ├── train_centralized.py
│       └── train_local.py
├── tests/
│   └── test_backend.py
├── checkpoint.md
├── README.md
└── requirements.txt
```

### Flutter app structure

The Flutter companion app is already implemented with the following structure:

```text
lib/
├── app.dart, main.dart
├── core/
├── models/
├── providers/
├── routes/
├── screens/
├── services/
└── widgets/
```

Key screens:
- Home
- Live Monitor
- History
- Settings
- Device
- Splash
- About

---

## Setup

### 1. Create virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Flutter dependencies

The Flutter app requires packages such as:
- `mqtt_client`
- `fl_chart`
- `flutter_local_notifications`
- `provider`
- `shared_preferences`
- `uuid`
- `http`
- `intl`

Then run:

```bash
flutter pub get
```

---

## Data Pipeline

### Step 1 — Place PAMAP2 raw files

```text
data/raw/subject101.dat
data/raw/subject102.dat
...
data/raw/subject109.dat
```

### Step 2 — Preprocess all subjects

```bash
python -m src.data.preprocess
```

This step:
- discards transient activity `0`
- interpolates missing heart-rate values
- maps activities into grouped intensity classes
- builds rolling/statistical features
- generates pseudo-labels

### Step 3 — Split into FL clients and test set

```bash
python -m src.data.split_clients
```

Split design:
- `subject101..105` → `client_1..client_5`
- `subject106..109` → global test set

### Client label distribution

| Client | Subject | OK (0) | MEDIUM (1) | HIGH (2) |
|---|---|---:|---:|---:|
| client_1 | 101 | 15.1% | 32.4% | 52.5% |
| client_2 | 102 | 48.4% | 29.9% | 21.7% |
| client_3 | 103 | 71.5% | 14.3% | 14.2% |
| client_4 | 104 | 27.5% | 35.9% | 36.6% |
| client_5 | 105 | 50.4% | 37.7% | 11.9% |

This naturally creates a non-IID federated learning setup.

---

## Training

### Centralized baseline

```bash
python -m src.training.train_centralized
```

Expected output:
- `models/centralized_mlp.pt`

### Local-only baseline

```bash
python -m src.training.train_local
```

Expected outputs:
- `models/local_client_1.pt` ... `models/local_client_5.pt`

### Federated Learning with Flower

#### Start server

```bash
python -m src.fl.server
```

#### Start 5 clients in separate terminals

```bash
python -m src.fl.client --cid 1 --server 127.0.0.1:8080
python -m src.fl.client --cid 2 --server 127.0.0.1:8080
python -m src.fl.client --cid 3 --server 127.0.0.1:8080
python -m src.fl.client --cid 4 --server 127.0.0.1:8080
python -m src.fl.client --cid 5 --server 127.0.0.1:8080
```

### FL configuration

| Parameter | Value |
|---|---|
| Rounds | 15 |
| Local epochs per round | 1 |
| Batch size | 512 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Aggregation | FedAvg |

These settings come from the original project pipeline and remain part of the training workflow.

---

## Evaluation

### Compare models

```bash
python -m src.evaluation.compare_results
```

### Generate plots

```bash
python -m src.evaluation.plot_results
```

Plots are saved under:

```text
experiments/plots/
```

### Main results

#### Centralized

| Metric | Score |
|---|---:|
| Accuracy | 0.999536 |
| Precision Macro | 0.999243 |
| Recall Macro | 0.999535 |
| F1 Macro | 0.999389 |

#### Local average

| Metric | Score |
|---|---:|
| Accuracy | 0.966997 |
| Precision Macro | 0.964467 |
| Recall Macro | 0.954392 |
| F1 Macro | 0.956561 |

#### Flower best

| Metric | Score |
|---|---:|
| Accuracy | 0.999659 |
| Precision Macro | 0.999473 |
| Recall Macro | 0.999626 |
| F1 Macro | 0.999550 |

#### Interpretation

- FL is nearly identical to centralized performance, with **F1 difference < 0.0002**
- FL significantly outperforms average local-only training by about **+0.043 macro F1**

This strongly supports the privacy-preserving federated learning design.

---

## Inference Backend

The FastAPI backend is complete and ready for real-time prediction.

### Main endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check server + model |
| `/labels` | GET | Label mapping |
| `/device-activity-map` | GET | Device activity mapping |
| `/predict` | POST | Predict from direct features |
| `/predict-from-sensor` | POST | Predict from raw sensor data |

### Start backend

```bash
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` is required if a real phone on the same WiFi network needs to access the backend.

### Request format for `/predict-from-sensor`

```json
{
  "heart_rate": 132.0,
  "hr_window": [126, 128, 130, 131, 132],
  "acc_x": 0.8,
  "acc_y": 9.6,
  "acc_z": 1.5,
  "device_activity_code": 2,
  "timestamp_seconds": 1234567890.0
}
```

### Important model files

- `models/flower_global_best.pt`
- `models/flower_scaler.pkl`

### Network note

When testing on a real phone:
- use your laptop LAN IP instead of `10.0.2.2`
- open Windows Firewall port `8000`

---

## SQLite Logger

The SQLite logger is implemented in `src/inference/logger.py` and supports:
- reading logs
- alert logs
- FL round logs
- FL client logs

Current note:
- logger is already used by `realtime_loop.py`
- for Option B, it should be integrated into `api_server.py` so predictions coming from Flutter are also logged

---

## Real-time Inference Paths

### 1. Primary flow — Option B

The main architecture now is:

```text
ESP32 → MQTT → Flutter → FastAPI → Flutter → MQTT → ESP32
```

This is the official demo path.

### 2. Fallback flow — Python serial loop

`src/inference/realtime_loop.py` still exists and works, but it is **no longer the primary demo architecture**. It is retained as:
- a fallback when the phone is unavailable
- a standalone testing tool
- a reporting/demo backup

Run fallback demo mode:

```bash
python src/inference/realtime_loop.py
```

Run serial mode:

```bash
python src/inference/realtime_loop.py --port COM5
```

---

## ESP32 Hardware Integration

### Basic hardware tasks

Implemented firmware components include:
- pulse sensor reading
- MPU6050 motion reading
- alert manager
- shared globals
- FreeRTOS task orchestration

### Option B firmware requirements

To fully support the MQTT-based architecture, the firmware still needs these updates:
- add `AlertLevel` enum in `globals.h/.cpp`
- switch MQTT server to `broker.hivemq.com`
- rewrite `comm_task.cpp`
- subscribe `wearable/alert`
- publish `wearable/data`
- handle 3-level alert behavior in `alert_manager.cpp`

### MQTT topics

Official topics are:
- `wearable/data` — ESP32 publishes, Flutter subscribes
- `wearable/alert` — Flutter publishes, ESP32 subscribes

### Broker

Primary:
```text
broker.hivemq.com:1883
```

Backup:
```text
test.mosquitto.org:1883
```

---

## Flutter Mobile App

The Flutter app supports:
- manual input prediction
- mock streaming
- real-time monitoring screen
- history storage
- local notifications
- real-device mode

### Important modes

Available app modes:
- `manual`
- `mock`
- `real_device`

### Config notes

Default app settings include:
- default backend URL: `http://10.0.2.2:8000`
- default predict interval: `3000 ms`
- HR window size: `5`

When testing on a real Android phone, update the backend URL to your laptop LAN IP in Settings.

### Remaining Flutter integration work

According to the latest checkpoint, the remaining core work is:
- MQTT-based `device_service.dart`
- public `deviceService` in `sensor_provider.dart`
- `PredictionProvider` sending `A/M/O` commands
- auto-predict timer in `LiveMonitorScreen`
- complete real-device end-to-end test

---

## Demo Checklist

### Preparation
- Flash ESP32 with updated firmware
- Confirm ESP32 subscribes to `wearable/alert`
- Start backend:
  ```bash
  uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000
  ```
- Test:
  ```text
  http://<LAPTOP_IP>:8000/health
  ```
- In Flutter Settings:
  - set Base URL to `http://<LAPTOP_IP>:8000`
  - click **Check Backend**
  - set mode to `real_device`[21]

### During demo
- Open **Live Monitor**
- Tap **Connect Device**
- Enable **Auto Predict**
- Observe:
  - real-time HR chart updates
  - prediction state changes (`OK` / `MEDIUM` / `HIGH`)
  - ESP32 LED / buzzer reacts
  - history entries are saved

### Fallback
If MQTT/WiFi is unstable:
- switch app mode to `mock`
- or run `realtime_loop.py` via serial

---

## Testing

A backend test file exists:

```text
tests/test_backend.py
```

You can use it to verify API behavior during development.

---

## Notes and Limitations

- The labels are pseudo-labels derived from heart rate and activity context, so the task is an engineered alert-classification problem rather than a clinically validated medical diagnosis.
- Current results are very high partly because the feature design strongly aligns with the rule used to generate labels.
- This project demonstrates the value of FL under non-IID data while keeping raw physiological data local to each client[38].
- The **current authoritative implementation details** should follow the latest checkpoint and codebase, not the old README where conflicts exist[38].

---

## Future Work

- Finish full MQTT-based ESP32 ↔ Flutter ↔ Backend ↔ ESP32 integration
- Integrate SQLite logging directly into `api_server.py`
- Export model for mobile inference / ONNX deployment
- Explore secure aggregation or differential privacy
- Test on larger real wearable datasets
- Extend from global FL to personalized FL fine-tuning

---

## References

- PAMAP2 Physical Activity Monitoring dataset
- PyTorch
- Flower Federated Learning framework
- Flutter
- FastAPI
- MQTT / HiveMQ
```
