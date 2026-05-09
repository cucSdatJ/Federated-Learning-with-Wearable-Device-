# FL Wearable

Federated Learning for Personalized Wearable Device Control using **PAMAP2**, **PyTorch**, and **Flower**.

## Overview

This project builds a privacy-preserving wearable health monitoring pipeline where multiple clients collaboratively train a global model **without sharing raw data**.

The model classifies heart rate readings into three alert levels — `OK`, `MEDIUM`, `HIGH` — based on heart rate and activity context. The trained global model is deployed to an ESP32-based wearable device for real-time inference and hardware alert triggering.

Main components:

- PAMAP2 data pipeline with activity-aware pseudo-labeling
- 3-class alert classification (`OK`, `MEDIUM`, `HIGH`)
- Centralized baseline training
- Local-only baseline (per-client)
- Federated Learning with Flower (server + 5 clients)
- Result comparison and plotting

---

## Dataset

We use the **PAMAP2 Physical Activity Monitoring** dataset.  
Download: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

Place raw files under `data/raw/`:

```
data/raw/
├── subject101.dat
├── subject102.dat
...
└── subject109.dat
```

### Activities present in this dataset

Out of 19 activity IDs described in the PAMAP2 README, only **13 IDs** are actually present across all 9 subjects:

| ID | Activity |
|---:|---|
| 0 | transient *(discarded)* |
| 1 | lying |
| 2 | sitting |
| 3 | standing |
| 4 | walking |
| 5 | running |
| 6 | cycling |
| 7 | nordic\_walking |
| 12 | ascending\_stairs |
| 13 | descending\_stairs |
| 16 | vacuum\_cleaning |
| 17 | ironing |
| 24 | rope\_jumping |

IDs `9, 10, 11, 18, 19, 20` are listed in the PAMAP2 README but **do not appear** in any of the 9 subject files used in this project.

### Dataset characteristics

Activity sample counts (all subjects combined):

| Activity | Count |
|---|---:|
| walking | 238,761 |
| ironing | 238,690 |
| nordic\_walking | 188,107 |
| cycling | 164,600 |
| vacuum\_cleaning | 175,353 |
| sitting | 185,188 |
| standing | 189,931 |
| lying | 192,523 |
| ascending\_stairs | 117,216 |
| descending\_stairs | 104,944 |
| running | 98,199 |
| rope\_jumping | 49,360 |

Mean heart rate by activity (all subjects, interpolated HR):

| Activity | Mean HR (bpm) |
|---|---:|
| lying | 75.54 |
| sitting | 80.01 |
| standing | 88.55 |
| ironing | 90.06 |
| vacuum\_cleaning | 104.20 |
| walking | 112.79 |
| nordic\_walking | 123.83 |
| cycling | 124.88 |
| ascending\_stairs | 129.53 |
| descending\_stairs | 129.16 |
| running | 156.59 |
| rope\_jumping | 161.99 |

Heart rate varies clearly and monotonically across activity intensity levels, which makes the pseudo-labeling strategy effective for model learning.

---

## Activity Grouping

Activities are grouped into 4 intensity levels based on observed mean HR, and used for both one-hot features and labeling thresholds:

| Group | Activity IDs | Mean HR range |
|---|---|---|
| `rest` | 1 (lying), 2 (sitting), 3 (standing), 17 (ironing) | 75 – 90 bpm |
| `walk` | 4 (walking), 16 (vacuum\_cleaning) | 104 – 113 bpm |
| `brisk` | 6 (cycling), 7 (nordic\_walking), 12 (ascending\_stairs), 13 (descending\_stairs) | 124 – 130 bpm |
| `run` | 5 (running), 24 (rope\_jumping) | 157 – 162 bpm |

> **Note:** `activity_id = 0` (transient) is always discarded before any processing.

---

## Labeling Strategy

Since PAMAP2 does not provide risk labels, we generate **pseudo-labels** using activity-aware heart-rate thresholds:

- `0 = OK`
- `1 = MEDIUM`
- `2 = HIGH`

Thresholds are set based on observed HR distributions per activity group:

| Group | MEDIUM (≥) | HIGH (≥) |
|---|---:|---:|
| rest | 90 | 100 |
| walk | 105 | 120 |
| brisk | 120 | 135 |
| run | 150 | 170 |

These thresholds are derived from actual HR statistics across all 9 subjects, not from clinical standards. The labeling converts the problem into a supervised 3-class classification task for local and federated training.

> See the Notes section for an important caveat about the nature of these pseudo-labels.

---

## Features

The model uses 10 input features, all computable from ESP32 sensor data:

| # | Feature | Source |
|---|---|---|
| 1 | `heart_rate` | Pulse Sensor (currentBPM) |
| 2 | `hr_rolling_mean` | Rolling mean over 10 samples |
| 3 | `hr_rolling_std` | Rolling std over 10 samples |
| 4 | `acc_magnitude` | `sqrt(x² + y² + z²)` from hand IMU (col 4,5,6) |
| 5 | `hour_sin` | `sin(2π × hour / 24)` — cyclical time encoding |
| 6 | `hour_cos` | `cos(2π × hour / 24)` — cyclical time encoding |
| 7 | `act_rest` | One-hot: activity group = rest |
| 8 | `act_walk` | One-hot: activity group = walk |
| 9 | `act_brisk` | One-hot: activity group = brisk |
| 10 | `act_run` | One-hot: activity group = run |

> **Important:** Accelerometer columns used are **col 4, 5, 6** (hand IMU), not col 10–12 (which are hand gyroscope).  
> This matches the placement of the wearable device on the wrist, consistent with ESP32 + MPU6050 hardware.

---

## Repository Structure

```
fl_wearable/
├── data/
│   ├── raw/                        # PAMAP2 .dat files (not committed)
│   └── processed/                  # Generated CSVs
│       ├── all_processed.csv
│       ├── class_distribution.csv
│       ├── client_1.csv .. client_5.csv
│       ├── test_set.csv
│       ├── activity_counts_all_subjects.csv
│       ├── activity_hr_stats_all_subjects.csv
│       └── activity_hr_stats_per_subject.csv
├── experiments/
│   ├── plots/
│   ├── ...
│   ├── local_results.csv
│   ├── compare_results.csv
│   └── compare_summary.csv
├── hardware/                        # ESP32 PlatformIO firmware
├── models/
│   ├── centralized_mlp.pt
│   ├── scaler.pkl                   # Centralized scaler
│   ├── local_client_1.pt .. local_client_5.pt
│   ├── scaler_client_1.pkl .. scaler_client_5.pkl
│   ├── fl_global_best.pt
│   └── ...
├── src/
│   ├── data/
│   │   ├── utils.py                 # map_activity(), label_3class(), acc_magnitude()
│   │   ├── pamap2_loader.py         # load_subject() → processed DataFrame
│   │   ├── preprocess.py            # Run all subjects → all_processed.csv
│   │   ├── split_clients.py         # Split → client_1..5.csv + test_set.csv
│   │   ├── dataset.py               # WearableDataset (PyTorch Dataset)
│   │   └── scaler.py                # fit_scaler(), transform_df()
│   ├── models/
│   │   └── mlp.py                   # MLPClassifier (10 → 64 → 32 → 3)
│   ├── training/
│   │   ├── train_centralized.py
│   │   ├── train_local.py
│   │   ├── compare_results.py
│   │   └── plot_results.py
│   └── fl/
│       ├── common.py                # Shared utilities for Flower
│       ├── client.py                # WearableFlowerClient
│       └── server.py                # Flower server with evaluate_fn
├── requirements.txt
└── README.md
```

---

## Setup

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Pipeline

### Step 1 — Place raw files

```
data/raw/subject101.dat
data/raw/subject102.dat
...
data/raw/subject109.dat
```

### Step 2 — Preprocess all subjects

```bash
python -m src.data.preprocess
```

Reads each `.dat` file, applies:
- discard `activity_id = 0` (transient)
- linear interpolation for NaN heart rate
- activity grouping via `map_activity()`
- feature extraction (rolling stats, acc magnitude, cyclical time, one-hot)
- pseudo-label generation via `label_3class()`

Outputs:
```
data/processed/all_processed.csv
data/processed/class_distribution.csv
```

### Step 3 — Split into clients and test set

```bash
python -m src.data.split_clients
```

Split:
- `subject101..105` → `client_1.csv` .. `client_5.csv` (training clients)
- `subject106..109` → `test_set.csv` (global evaluation)

Outputs:
```
data/processed/
├── client_1.csv     (248,599 rows)
├── client_2.csv     (260,902 rows)
├── client_3.csv     (173,851 rows)
├── client_4.csv     (229,262 rows)
├── client_5.csv     (271,021 rows)
└── test_set.csv     (748,113 rows)
```

Each CSV has 13 columns: `subject_id`, `timestamp`, 10 features, `label`.

### Client label distribution (training set)

| Client | Subject | OK (0) | MEDIUM (1) | HIGH (2) |
|---|---|---:|---:|---:|
| client\_1 | 101 | 15.1% | 32.4% | 52.5% |
| client\_2 | 102 | 48.4% | 29.9% | 21.7% |
| client\_3 | 103 | 71.5% | 14.3% | 14.2% |
| client\_4 | 104 | 27.5% | 35.9% | 36.6% |
| client\_5 | 105 | 50.4% | 37.7% | 11.9% |

Each client has all 3 classes present. The distribution varies naturally between subjects, reflecting genuine non-IID data heterogeneity — a key property for realistic federated learning evaluation.

---

## Training

### Centralized baseline

Combines all 5 client CSVs into a single training set, fits a StandardScaler, and trains one MLP:

```bash
python -m src.training.train_centralized
```

Outputs: `models/centralized_mlp.pt`, `models/scaler.pkl`

### Local-only baseline

Trains a separate MLP for each client independently, each with its own scaler and class weights:

```bash
python -m src.training.train_local
```

Outputs: `models/local_client_1.pt` .. `models/local_client_5.pt`, `models/scaler_client_1.pkl` .. `models/scaler_client_5.pkl`

Average local performance is used as the `local_avg` baseline in the final comparison.

### Flower Federated Learning

#### Start server

```bash
python -m src.fl.server
```

The server:
- uses `FedAvg` strategy
- evaluates the global model on `test_set.csv` after each round
- saves the best global model as `models/fl_global_best.pt`
- logs per-round metrics to `experiments/fl_round_metrics.csv`

#### Start 5 clients (in separate terminals)

```bash
python -m src.fl.client --cid 1 --server 127.0.0.1:8080
python -m src.fl.client --cid 2 --server 127.0.0.1:8080
python -m src.fl.client --cid 3 --server 127.0.0.1:8080
python -m src.fl.client --cid 4 --server 127.0.0.1:8080
python -m src.fl.client --cid 5 --server 127.0.0.1:8080
```

Each client:
- loads its own `client_N.csv`
- trains locally for `LOCAL_EPOCHS` (default: 1) per round
- uses `CrossEntropyLoss` with per-client class weights (balanced)
- exchanges only model weights with the server — raw data never leaves the client

#### FL configuration

| Parameter | Value |
|---|---|
| Rounds | 15 |
| Local epochs per round | 1 |
| Batch size | 512 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Aggregation | FedAvg (weighted by number of samples) |

---

## Evaluation

Compare all settings and generate plots:

```bash
python -m src.training.compare_results
python -m src.training.plot_results
```

Plots saved to `experiments/plots/`.

---

## Current Main Results

| Setting | Accuracy | Precision Macro | Recall Macro | F1 Macro |
|---|---:|---:|---:|---:|
| centralized | 0.9992 | 0.9989 | 0.9994 | 0.9991 |
| flower\_best | 0.9985 | 0.9972 | 0.9982 | 0.9977 |
| local\_avg | 0.9474 | 0.9326 | 0.9385 | 0.9302 |

### Flower FL convergence (selected rounds)

| Round | Avg Local Loss | Accuracy | F1 Macro |
|---:|---:|---:|---:|
| 1 | 1.1525 | 0.8841 | 0.8451 |
| 5 | 0.0541 | 0.9901 | 0.9857 |
| 10 | 0.0142 | 0.9970 | 0.9951 |
| 15 | 0.0056 | 0.9985 | 0.9977 |

### Summary

- **Centralized** performs best as expected (upper bound).
- **Flower FL** achieves near-centralized performance (F1 difference < 0.002), while keeping all raw data local to each client.
- **Flower FL** clearly outperforms average local-only training (F1 improvement: +0.068 macro F1).
- FL convergence is stable: macro F1 rises from 0.845 in round 1 to 0.998 by round 15.

---

## Model Architecture

```
MLPClassifier(
  Linear(10, 64) → BatchNorm(64) → ReLU → Dropout(0.3)
  Linear(64, 32) → BatchNorm(32) → ReLU → Dropout(0.2)
  Linear(32, 16) → ReLU
  Linear(16, 3)
)
```

- Input: 10 features
- Output: 3 logits (no softmax — CrossEntropyLoss handles it)
- Loss: `CrossEntropyLoss` with balanced class weights computed per client
- Optimizer: Adam, lr = 1e-3

---

## Hardware Integration

The trained global model (`models/fl_global_best.pt`) is deployed to a gateway laptop connected via USB Serial to an ESP32 wearable device.

### Data flow

```
ESP32 (sensor reading)
  → Serial: "BPM,Activity\n"
  → Python gateway (realtime_loop.py)
    → load global_model.pt
    → 3-class inference
    → Serial command back to ESP32:
        'A' = HIGH alert   → LED red + Buzzer
        'M' = MEDIUM alert → LED slow blink
        'O' = OK           → LED off, silent
```

### ESP32 firmware (FreeRTOS tasks)

| Task | Description |
|---|---|
| `TaskReadPulse` | Reads Pulse Sensor (pin 32), peak detection, computes BPM |
| `TaskReadMotion` | Reads MPU6050 via I2C, classifies activity from acc magnitude |
| `TaskCommunicate` | Sends `BPM,Activity` via Serial; receives `A/M/O` commands; publishes MQTT to Core IoT |
| `TaskAlertManager` | Controls Buzzer (pin 27) and Red LED (pin 2) based on `alertLevel` |

---

## Notes

- Results are high because the pseudo-labels are derived directly from heart rate and activity context, while the input features also include heart-rate statistics and activity indicators. The model is effectively learning an engineered rule, not a clinically validated diagnostic pattern.
- This project demonstrates effective federated learning of a wearable alert classification task, with the key property that **raw physiological data never leaves the local device**.
- For clinical or medical use, labels would need to be replaced with verified ground-truth annotations.
- The PAMAP2 dataset only contains 13 of 19 documented activity IDs across the 9 subjects used. Activities `9, 10, 11, 18, 19, 20` are absent.
- Heart rate column in PAMAP2 is sampled at 4 Hz (vs. 100 Hz for IMU), resulting in many NaN rows. Linear interpolation is applied per subject before feature extraction.

---

## Future Work

- Export best FL model to ONNX for mobile inference
- Integrate model with Flutter app (real-time BPM + alert display)
- Add secure aggregation or differential privacy to the FL pipeline
- Test on real wearable data collected from ESP32 hardware
- Explore personalized federated learning (per-client fine-tuning after global aggregation)

