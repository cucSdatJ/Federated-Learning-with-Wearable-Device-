# CHECKPOINT — FL Wearable Project
> Cập nhật tới: 09/05/2026

---

## Tổng quan nhanh

| Module | Trạng thái | Tiến độ |
|---|---|---|
| ESP32 Firmware | ✅ Hoàn thành | 100% |
| PAMAP2 Data Pipeline | ✅ Hoàn thành | 100% |
| MLP Model (3-class) | ✅ Hoàn thành | 100% |
| Centralized Training | ✅ Hoàn thành | 100% |
| Local-only Training | ✅ Hoàn thành | 100% |
| Federated Learning (Flower) | ✅ Hoàn thành | 100% |
| Evaluation & Plots | ✅ Hoàn thành | 100% |
| README.md | ✅ Hoàn thành | 100% |
| Flutter Mobile App | ❌ Chưa bắt đầu | 0% |
| Real-time Inference Loop | ⚠️ Cần update | 60% |
| ESP32 MEDIUM alert (`'M'`) | ⚠️ Cần update | 70% |
| SQLite Database | ❌ Chưa bắt đầu | 0% |
| Adafruit IO Dashboard | ❌ Chưa bắt đầu | 0% |
| Báo cáo cuối kỳ | ⚠️ Đang làm | 25% |

---

## 1. Hardware — ESP32 Firmware ✅

### Đã hoàn thành
- `main.cpp` — khởi tạo FreeRTOS, tạo 4 tasks, init pins
- `pulse_sensor.cpp` — `TaskReadPulse`: đọc analog pin 32, peak detection, tính BPM, filter 30–200 BPM, moving average
- `motion_sensor.cpp` — `TaskReadMotion`: đọc MPU6050 qua I2C, tính acc magnitude, classify activity (0=rest, 1=walk, 2=brisk, 4=run)
- `comm_task.cpp` — `TaskCommunicate`: gửi `BPM,Activity\n` qua Serial 115200, nhận lệnh `'A'`/`'O'`, publish MQTT lên Core IoT
- `alert_manager.cpp` — `TaskAlertManager`: LED P2 + Buzzer P27 nhấp nháy khi `alertTriggered=true`
- `config.h` — pin definitions, WiFi SSID/Pass, MQTT server/token
- `globals.h/.cpp` — shared variables: `currentBPM`, `currentActivity`, `alertTriggered`

### Đã test thực tế
- Kết nối ESP32 qua COM5 thành công
- Python inference loop đọc được `BPM,Activity` từ Serial
- Gửi lệnh `'A'` → Buzzer + LED kích hoạt đúng
- MQTT publish lên Core IoT hoạt động

### Cần làm (nhỏ)
- [ ] Thêm lệnh `'M'` (MEDIUM alert): nhận ở `comm_task.cpp`, LED nhấp nháy chậm ở `alert_manager.cpp`
- [ ] Thêm `AlertLevel` enum vào `globals.h` (`ALERT_OK`, `ALERT_MED`, `ALERT_HIGH`)
- [ ] Comment out dòng hardcode test trong `pulse_sensor.cpp`:
  ```cpp
  // if (currentBPM > 120) alertTriggered = true; else alertTriggered = false;
  ```
  (Python AI mới là bên quyết định alert)

---

## 2. PAMAP2 Data Pipeline ✅

### Đã hoàn thành

#### 2.1 Khám phá dataset
- Xác minh 9 file `.dat`: `subject101.dat` → `subject109.dat`
- Chỉ **13/19** activity IDs thực sự xuất hiện trong dataset này (IDs `9, 10, 11, 18, 19, 20` không có)
- Đã thống kê activity counts toàn bộ dataset:

| Activity | Count |
|---|---:|
| walking (4) | 238,761 |
| ironing (17) | 238,690 |
| nordic_walking (7) | 188,107 |
| cycling (6) | 164,600 |
| vacuum_cleaning (16) | 175,353 |
| ascending_stairs (12) | 117,216 |
| running (5) | 98,199 |
| descending_stairs (13) | 104,944 |
| rope_jumping (24) | 49,360 |

- Đã thống kê HR mean/median/std theo activity trên toàn 9 subjects:

| Activity | Mean HR | Nhóm |
|---|---:|---|
| lying (1) | 75.54 | rest |
| sitting (2) | 80.01 | rest |
| standing (3) | 88.55 | rest |
| ironing (17) | 90.06 | rest |
| vacuum_cleaning (16) | 104.20 | walk |
| walking (4) | 112.79 | walk |
| nordic_walking (7) | 123.83 | brisk |
| cycling (6) | 124.88 | brisk |
| descending_stairs (13) | 129.16 | brisk |
| ascending_stairs (12) | 129.53 | brisk |
| running (5) | 156.59 | run |
| rope_jumping (24) | 161.99 | run |

#### 2.2 Activity mapping (chốt cuối)
Grouping dựa trên HR thực tế, không phải tên activity:
```python
rest  = [1, 2, 3, 17]      # HR mean: 75–90
walk  = [4, 16]             # HR mean: 104–113
brisk = [6, 7, 12, 13]     # HR mean: 124–130
run   = [5, 24]             # HR mean: 157–162
```
> Note: `cycling (6)` → `brisk` (không phải `run`) vì HR mean 124.88 gần stairs/nordic, không gần running 156.

#### 2.3 Pseudo-labeling (chốt cuối)
Threshold dựa trên HR thực tế mỗi nhóm:
```
rest:  MEDIUM ≥ 90,  HIGH ≥ 100
walk:  MEDIUM ≥ 105, HIGH ≥ 120
brisk: MEDIUM ≥ 120, HIGH ≥ 135
run:   MEDIUM ≥ 150, HIGH ≥ 170
```

#### 2.4 Feature engineering (10 features, match ESP32)
```
heart_rate, hr_rolling_mean, hr_rolling_std,
acc_magnitude (cột 4,5,6 — hand IMU, KHÔNG phải chest),
hour_sin, hour_cos,
act_rest, act_walk, act_brisk, act_run
```

#### 2.5 Files đã generate
```
data/processed/
├── all_processed.csv
├── class_distribution.csv
├── client_1.csv     (subject101 — 248,599 rows)
├── client_2.csv     (subject102 — 260,902 rows)
├── client_3.csv     (subject103 — 173,851 rows)
├── client_4.csv     (subject104 — 229,262 rows)
├── client_5.csv     (subject105 — 271,021 rows)
├── test_set.csv     (subject106–109 — 748,113 rows)
├── activity_counts_all_subjects.csv
├── activity_hr_stats_all_subjects.csv
└── activity_hr_stats_per_subject.csv
```

#### 2.6 Label distribution mỗi client (đã verify — không còn NaN)
| Client | Subject | OK (0) | MEDIUM (1) | HIGH (2) |
|---|---|---:|---:|---:|
| client_1 | 101 | 15.1% | 32.4% | 52.5% |
| client_2 | 102 | 48.4% | 29.9% | 21.7% |
| client_3 | 103 | 71.5% | 14.3% | 14.2% |
| client_4 | 104 | 27.5% | 35.9% | 36.6% |
| client_5 | 105 | 50.4% | 37.7% | 11.9% |
| test_set | 106–109 | ~70% | ~14% | ~16% |

Non-IID tự nhiên giữa các subject — đúng tinh thần FL.

#### 2.7 Scripts pipeline
```bash
python -m src.data.preprocess      # → all_processed.csv
python -m src.data.split_clients   # → client_1..5.csv + test_set.csv
```

---

## 3. Model MLP ✅

### Đã hoàn thành
- `src/models/mlp.py` — `MLPClassifier`:
  ```
  Linear(10, 64) → BatchNorm → ReLU → Dropout(0.3)
  Linear(64, 32) → BatchNorm → ReLU → Dropout(0.2)
  Linear(32, 16) → ReLU
  Linear(16, 3)   ← 3 outputs: OK / MEDIUM / HIGH
  ```
- Loss: `CrossEntropyLoss` với balanced class weights per client
- Optimizer: Adam, lr = 1e-3

---

## 4. Centralized Training ✅

### Đã hoàn thành
- `src/training/train_centralized.py`
- Gộp client_1..5 → train set, fit StandardScaler, train MLP
- Evaluate trên `test_set.csv`

### Kết quả
| Metric | Score |
|---|---:|
| Accuracy | 0.9992 |
| Precision Macro | 0.9989 |
| Recall Macro | 0.9994 |
| F1 Macro | 0.9991 |

### Files output
```
models/centralized_mlp.pt
models/scaler.pkl
experiments/centralized_metrics.json
experiments/centralized_classification_report.txt
experiments/centralized_history.csv
```

---

## 5. Local-only Training ✅

### Đã hoàn thành
- `src/training/train_local.py`
- Train riêng từng client (5 models độc lập)
- Mỗi client có scaler riêng, class weights riêng

### Files output
```
models/local_client_1.pt .. local_client_5.pt
models/scaler_client_1.pkl .. scaler_client_5.pkl
experiments/local_results.csv
experiments/client_*_history.csv
experiments/client_*_val_metrics.json
experiments/client_*_test_metrics.json
```

### Kết quả trung bình (local_avg)
| Metric | Score |
|---|---:|
| Accuracy | 0.9474 |
| Precision Macro | 0.9326 |
| Recall Macro | 0.9385 |
| F1 Macro | 0.9302 |

---

## 6. Federated Learning — Flower ✅

### Đã hoàn thành

#### 6.1 Architecture
- `src/fl/common.py` — shared utils: `load_client_data()`, `df_to_loader()`, `get_model()`, `get_parameters()`, `set_parameters()`, `train_model()`, `evaluate_model()`, `compute_local_class_weights()`
- `src/fl/client.py` — `WearableFlowerClient(NumPyClient)`: `fit()` + `evaluate()`
- `src/fl/server.py` — Flower server với `get_evaluate_fn()` evaluate global model trên `test_set.csv` sau mỗi round

#### 6.2 Config
| Parameter | Value |
|---|---|
| Rounds | 15 |
| Local epochs/round | 1 |
| Batch size | 512 |
| Learning rate | 1e-3 |
| Aggregation | FedAvg (weighted by samples) |
| Class weights | Balanced per client |

#### 6.3 FL convergence (Flower)
| Round | Avg Local Loss | Accuracy | F1 Macro |
|---:|---:|---:|---:|
| 0 (init) | — | 0.1630 | 0.0944 |
| 1 | 1.1525 | 0.8841 | 0.8451 |
| 5 | 0.0541 | 0.9901 | 0.9857 |
| 10 | 0.0142 | 0.9970 | 0.9951 |
| 15 | 0.0056 | 0.9985 | 0.9977 |

#### 6.4 Kết quả best (Flower)
| Metric | Score |
|---|---:|
| Accuracy | 0.9985 |
| Precision Macro | 0.9972 |
| Recall Macro | 0.9982 |
| F1 Macro | 0.9977 |

#### 6.5 Files output
```
models/fl_global_best.pt
models/fl_scaler.pkl
experiments/fl_round_metrics.csv
experiments/fl_best_metrics.json
experiments/fl_best_classification_report.txt
```

#### 6.6 Cách chạy
```bash
# Terminal 1 — Server
python -m src.fl.server

# Terminal 2–6 — 5 Clients
python -m src.fl.client --cid 1 --server 127.0.0.1:8080
python -m src.fl.client --cid 2 --server 127.0.0.1:8080
python -m src.fl.client --cid 3 --server 127.0.0.1:8080
python -m src.fl.client --cid 4 --server 127.0.0.1:8080
python -m src.fl.client --cid 5 --server 127.0.0.1:8080
```

---

## 7. Evaluation & Comparison ✅

### So sánh 3 cấu hình chính
| Setting | Accuracy | Precision Macro | Recall Macro | F1 Macro |
|---|---:|---:|---:|---:|
| centralized | 0.9992 | 0.9989 | 0.9994 | **0.9991** |
| flower_best | 0.9985 | 0.9972 | 0.9982 | **0.9977** |
| local_avg | 0.9474 | 0.9326 | 0.9385 | **0.9302** |

### Kết luận học thuật
- Centralized = upper bound (toàn bộ data cùng một chỗ)
- **Flower FL chênh centralized < 0.002 F1** — gần như tương đương mà không cần share raw data
- **Flower FL vượt local_avg +0.068 F1** — đây là bằng chứng chính của giá trị FL

### Scripts đã có
```bash
python -m src.training.compare_results   # → compare_results.csv, compare_summary.csv
python -m src.training.plot_results      # → experiments/plots/*.png
```

### Plots đã generate
```
experiments/plots/
├── compare_summary_f1_macro.png
├── compare_summary_accuracy.png
├── compare_all_top10_f1_macro.png
├── flower_round_metrics.png
├── flower_round_loss.png
├── activity_counts.png
└── activity_hr_means.png
```

---

## 8. Documentation ✅

### Đã hoàn thành
- `README.md` — đầy đủ: dataset, features, activity grouping, labeling strategy, repo structure, setup, pipeline, training, FL, results, notes, future work

---

## 9. Real-time Inference Loop ⚠️ (cần update)

### Trạng thái hiện tại
- `inference/realtime_loop.py` đang chạy được với model cũ (binary/threshold)
- Đã test thực tế 20/4/2026 — kết nối COM5, detect HIGH alert đúng

### Terminal output thực tế (20/4/2026)
```
[*] Successfully connected to Wearable Hardware!
[*] Model successfully loaded from results/global_model.pth
21:41:23  140.0  rest  99.1%  [HIGH]   ← alert đúng
21:41:52  121.0  walk  79.9%  [MEDIUM]
21:42:04   92.0  rest   0.4%  OK
```

### Cần làm
- [ ] Update `realtime_loop.py` load model mới (`fl_global_best.pt` hoặc `fl_scaler.pkl`)
- [ ] Sửa inference: softmax 3-class → `argmax` → gửi `'A'`/`'M'`/`'O'` về ESP32
- [ ] Sửa feature extraction: đảm bảo 10 features đúng thứ tự, đúng cột acc hand (col 4,5,6)
- [ ] Test end-to-end lại với model Flower

```python
# Sau update:
label, conf = predict(model, features)
if label == 2:   ser.write(b'A')   # HIGH
elif label == 1: ser.write(b'M')   # MEDIUM (mới)
else:            ser.write(b'O')   # OK
```

---

## 10. Flutter Mobile App ❌ (chưa bắt đầu)

### Cần làm toàn bộ
- [ ] Setup Flutter project: `flutter create fl_wearable_app`
- [ ] `pubspec.yaml`: thêm `mqtt_client`, `fl_chart`, `flutter_local_notifications`, `provider`, `sqflite`
- [ ] `HomeScreen` — HR gauge + activity badge + risk badge (OK/MED/HIGH màu xanh/vàng/đỏ)
- [ ] `AlertScreen` — ListView alerts với timestamp, severity, HR value
- [ ] `HistoryScreen` — LineChart HR theo thời gian
- [ ] `SettingsScreen` — MQTT server, notification toggle
- [ ] `mqtt_service.dart` — kết nối Core IoT, subscribe telemetry
- [ ] `notification_service.dart` — push notification khi risk == 2

### MQTT data format (từ Core IoT)
```json
{"bpm": 85, "activity": 1, "risk": 2}
```

---

## 11. SQLite Database ❌ (chưa bắt đầu)

### Schema cần implement
```sql
-- 4 bảng
CREATE TABLE readings (id, timestamp, client_id, hr_bpm, activity, risk_prob, alert, source);
CREATE TABLE alerts   (id, timestamp, client_id, hr_bpm, severity, resolved, resolved_at);
CREATE TABLE fl_rounds  (round_id, timestamp, n_clients, avg_accuracy, avg_f1_macro);
CREATE TABLE fl_clients (id, round_id, client_id, local_f1, fl_f1, n_samples);
```

### Cần viết
- [ ] `database/schema.sql`
- [ ] `gateway/module4_logger.py`:
  - `init_db()`
  - `log_reading()`
  - `log_alert()`
  - `log_fl_round()`

---

## 12. Adafruit IO Dashboard ❌ (chưa bắt đầu)

### Feeds cần tạo
- `heart-rate` — HR gauge real-time
- `activity-level` — activity indicator (rest/walk/brisk/run)
- `alert-cmd` — alert level (OK/MEDIUM/HIGH)
- `fl-accuracy` — line chart FL accuracy per round [bonus HTTT]

### Cần làm
- [ ] `gateway/module1_display.py` — publish lên Adafruit feeds
- [ ] Tạo Adafruit IO account + feeds + dashboard widgets

---

## 13. Báo cáo cuối kỳ ⚠️ (đang làm)

### Đã có
- Results & Discussion section hoàn chỉnh (dựa trên số liệu thật)
- README đầy đủ để tham chiếu khi viết chương cơ sở lý thuyết và hiện thực

### Cần hoàn thiện
- [ ] Chương 1: Giới thiệu đề tài
- [ ] Chương 2: Cơ sở lý thuyết (IoT, FL, PAMAP2, MLP, Flutter, MQTT)
- [ ] Chương 3: Thiết kế hệ thống (kiến trúc 4 tầng, use-case, ER diagram, UI/UX)
- [ ] Chương 4: Hiện thực (ESP32 firmware, Strategy Pattern threshold, SQLite, FL pipeline)
- [ ] **Chương 5: Kết quả thực nghiệm** ← đã có data, cần viết thành văn
- [ ] Chương 6: Privacy Discussion (FL vs centralized, gradient leakage, GDPR)
- [ ] Chương 7: Kết luận & Hướng phát triển
- [ ] Phụ lục: Code snippets, screenshots

---

## 14. Tóm tắt công việc còn lại (ưu tiên)

### Ưu tiên 1 — Phần cứng (CE-1/CE-2) — nhỏ, nhanh
```
[ ] Thêm 'M' command vào comm_task.cpp + alert_manager.cpp
[ ] Test lại end-to-end alert 3 level với ESP32 thật
```

### Ưu tiên 2 — Inference update (CS-4) — trung bình
```
[ ] Update realtime_loop.py: load fl_global_best.pt, softmax 3-class, gửi A/M/O
[ ] Test với ESP32 thật
```

### Ưu tiên 3 — Database (CS-1/CE-3) — trung bình
```
[ ] Viết schema.sql + module4_logger.py
[ ] Tích hợp vào inference loop (log mỗi reading)
```

### Ưu tiên 4 — Flutter App (CE-3/CE-4) — lớn, cần sprint
```
[ ] Setup project + 3 screens + MQTT service + push notification
```

### Ưu tiên 5 — Dashboard (CE-4) — nhỏ
```
[ ] Tạo Adafruit feeds + module1_display.py
```

### Ưu tiên 6 — Báo cáo (CE-3/CE-4 + tất cả) — lớn
```
[ ] Viết 7 chương theo phân công, target ≥ 30 trang
```

---

## 15. Files quan trọng hiện có trong repo

```
fl_wearable/
├── hardware/src/
│   ├── main.cpp              ✅
│   ├── pulse_sensor.cpp      ✅
│   ├── motion_sensor.cpp     ✅
│   ├── comm_task.cpp         ⚠️ cần thêm 'M'
│   ├── alert_manager.cpp     ⚠️ cần thêm MEDIUM mode
│   ├── config.h              ✅
│   └── globals.h/cpp         ⚠️ cần AlertLevel enum
│
├── data/processed/
│   ├── client_1..5.csv       ✅
│   ├── test_set.csv          ✅
│   └── activity_*.csv        ✅
│
├── src/
│   ├── data/utils.py         ✅
│   ├── data/pamap2_loader.py ✅
│   ├── data/preprocess.py    ✅
│   ├── data/split_clients.py ✅
│   ├── data/dataset.py       ✅
│   ├── data/scaler.py        ✅
│   ├── models/mlp.py         ✅
│   ├── training/train_centralized.py ✅
│   ├── training/train_local.py       ✅
│   ├── training/compare_results.py   ✅
│   ├── training/plot_results.py      ✅
│   ├── fl/common.py          ✅
│   ├── fl/client.py          ✅
│   └── fl/server.py          ✅
│
├── models/
│   ├── centralized_mlp.pt    ✅
│   ├── scaler.pkl            ✅
│   ├── local_client_1..5.pt  ✅
│   ├── fl_global_best.pt     ✅
│   └── fl_scaler.pkl         ✅
│
├── experiments/
│   ├── compare_summary.csv   ✅
│   ├── fl_round_metrics.csv  ✅
│   └── plots/*.png           ✅
│
├── inference/realtime_loop.py ⚠️ cần update 3-class
│
├── flutter_app/               ❌ chưa có
├── database/schema.sql        ❌ chưa có
├── gateway/module*.py         ❌ chưa có
│
├── README.md                  ✅
└── requirements.txt           ✅
```

---

*Checkpoint này phản ánh trạng thái dự án tính đến 09/05/2026. Phần CS pipeline (data → FL → evaluation) đã hoàn thiện. Phần còn lại tập trung vào tích hợp phần cứng, mobile app, và báo cáo.*