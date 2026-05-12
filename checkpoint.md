# CHECKPOINT — FL Wearable Project
> Cập nhật tới: 12/05/2026

---

## Tổng quan nhanh

| Module | Trạng thái | Tiến độ |
|---|---|---|
| ESP32 Firmware (cơ bản) | ✅ Hoàn thành | 100% |
| ESP32 Firmware (Option B integration) | ⚠️ Đang sửa | 70% |
| PAMAP2 Data Pipeline | ✅ Hoàn thành | 100% |
| MLP Model (3-class) | ✅ Hoàn thành | 100% |
| Centralized Training | ✅ Hoàn thành | 100% |
| Local-only Training | ✅ Hoàn thành | 100% |
| Federated Learning (Flower) | ✅ Hoàn thành | 100% |
| Evaluation & Plots | ✅ Hoàn thành | 100% |
| FastAPI Inference Backend | ✅ Hoàn thành | 100% |
| SQLite Logger | ✅ Hoàn thành | 100% |
| Real-time Inference Loop (Serial) | ✅ Hoàn thành | 100% |
| Flutter Mobile App (cấu trúc) | ✅ Hoàn thành | 90% |
| Flutter ↔ ESP32 Integration (Option B) | ⚠️ Đang implement | 40% |
| Adafruit IO Dashboard | ❌ Chưa bắt đầu | 0% |
| README.md | ✅ Hoàn thành | 100% |
| Báo cáo cuối kỳ | ⚠️ Đang làm | 25% |

---

## Kiến trúc hệ thống đã chốt (Option B)

Kiến trúc end-to-end chính thức, **Flutter App làm trung gian** thay vì Python Serial loop:

```
ESP32 (Sensors)
  └── WiFi → MQTT publish → broker.hivemq.com:1883
                                    │
                        topic: wearable/data  {"bpm":92,"activity":1}
                                    │
                             Flutter App
                          (SensorProvider)
                                    │
                        POST /predict-from-sensor
                                    │
                         FastAPI Backend (Laptop)
                         flower_global_best.pt
                                    │
                         response: OK / MEDIUM / HIGH
                                    │
                             Flutter App
                          (PredictionProvider)
                                    │
                        topic: wearable/alert  "A"|"M"|"O"
                                    │
                             broker.hivemq.com
                                    │
                   ESP32 subscribe → mqttCallback → AlertLevel
                   → TaskAlertManager → LED P2 + Buzzer P27
```

**Hai MQTT topic cố định:**
- `wearable/data` — ESP32 publish, Flutter subscribe
- `wearable/alert` — Flutter publish, ESP32 subscribe

**Broker:** `broker.hivemq.com:1883` (public free, không cần tài khoản)
**Backup broker:** `test.mosquitto.org:1883`

---

## 1. Hardware — ESP32 Firmware

### 1.1 Đã hoàn thành (trước 09/05)
- `main.cpp` — khởi tạo FreeRTOS, tạo 4 tasks, init pins
- `pulse_sensor.cpp` — `TaskReadPulse`: đọc analog pin 32, peak detection, tính BPM, filter 30–200 BPM, moving average
- `motion_sensor.cpp` — `TaskReadMotion`: đọc MPU6050 qua I2C, tính acc magnitude, classify activity (0=rest, 1=walk, 2=brisk, 4=run)
- `alert_manager.cpp` — `TaskAlertManager`: LED P2 + Buzzer P27
- `config.h` — pin definitions, WiFi, MQTT config
- `globals.h/.cpp` — shared variables

### 1.2 Đã test thực tế (20/04/2026)
- Kết nối ESP32 qua COM5 thành công
- Python Serial loop đọc `BPM,Activity` đúng
- Lệnh `'A'` → Buzzer + LED kích hoạt đúng
- MQTT publish lên Core IoT hoạt động

### 1.3 Cần sửa cho Option B (⚠️ chưa làm)

**`globals.h`** — thêm AlertLevel enum:
```cpp
#pragma once
enum AlertLevel { ALERT_OK, ALERT_MED, ALERT_HIGH };
extern volatile int currentBPM;
extern volatile int currentActivity;
extern volatile bool alertTriggered;
extern volatile AlertLevel alertLevel;   // MỚI
```

**`globals.cpp`** — thêm init:
```cpp
volatile AlertLevel alertLevel = ALERT_OK;
```

**`config.h`** — đổi MQTT_SERVER sang HiveMQ:
```cpp
#define MQTT_SERVER "broker.hivemq.com"
#define MQTT_PORT   1883
// Giữ MQTT_TOKEN cho Core IoT nếu cần publish song song
```

**`comm_task.cpp`** — toàn bộ rewrite:
- Bỏ Serial loop (Python không còn làm trung gian)
- Thêm `mqttCallback()` nhận lệnh `'A'`/`'M'`/`'O'` từ Flutter
- Subscribe `wearable/alert` sau khi connect
- Publish `wearable/data` thay vì `v1/devices/me/telemetry`
- Client ID dùng MAC address để tránh conflict: `"ESP32_" + WiFi.macAddress()`

**`alert_manager.cpp`** — 3 mức dựa trên `alertLevel`:
```
ALERT_HIGH → LED + Buzzer nhấp nháy nhanh 200ms
ALERT_MED  → chỉ LED nhấp nháy chậm 700ms, Buzzer tắt
ALERT_OK   → tất cả tắt
```

**`pulse_sensor.cpp`** — bỏ hardcode test:
```cpp
// XÓA dòng này:
// if (currentBPM > 120) alertTriggered = true; else alertTriggered = false;
```

---

## 2. PAMAP2 Data Pipeline ✅

### Đã hoàn thành và chốt

#### Activity mapping
```python
rest  = [1, 2, 3, 17]   # HR mean 75–90 bpm
walk  = [4, 16]          # HR mean 104–113 bpm
brisk = [6, 7, 12, 13]  # HR mean 124–130 bpm
run   = [5, 24]          # HR mean 157–162 bpm
```

#### Pseudo-labeling thresholds
```
rest:  MEDIUM ≥ 90,  HIGH ≥ 100
walk:  MEDIUM ≥ 105, HIGH ≥ 120
brisk: MEDIUM ≥ 120, HIGH ≥ 135
run:   MEDIUM ≥ 150, HIGH ≥ 170
```

#### 8 features (chốt — bỏ hour_sin/hour_cos so với README cũ)
```
heart_rate, hr_rolling_mean, hr_rolling_std, acc_magnitude,
act_rest, act_walk, act_brisk, act_run
```
> Lưu ý: `src/models/mlp.py` dùng `input_dim=8`, không phải 10. README cũ ghi 10 features nhưng code thực tế là 8 (không có hour_sin/hour_cos).

#### Label distribution mỗi client
| Client | Subject | OK (0) | MEDIUM (1) | HIGH (2) |
|---|---|---:|---:|---:|
| client_1 | 101 | 15.1% | 32.4% | 52.5% |
| client_2 | 102 | 48.4% | 29.9% | 21.7% |
| client_3 | 103 | 71.5% | 14.3% | 14.2% |
| client_4 | 104 | 27.5% | 35.9% | 36.6% |
| client_5 | 105 | 50.4% | 37.7% | 11.9% |
| test_set | 106–109 | ~70% | ~14% | ~16% |

Non-IID tự nhiên — đúng tinh thần FL.

---

## 3. Model MLP ✅

```
MLPClassifier(input_dim=8, num_classes=3):
  Linear(8, 32) → ReLU → Dropout(0.2)
  Linear(32, 16) → ReLU → Dropout(0.1)
  Linear(16, 3)
```

> Lưu ý: Architecture trong `mlp.py` thực tế **nhỏ hơn** README mô tả (không có BatchNorm, không có layer 64). README ghi sai — code là nguồn đúng.

---

## 4. Centralized Training ✅

Kết quả mô hình centralized:

| Metric             | Score     |
| ------------------ | --------- |
| Accuracy           | 0.999536  |
| Precision Macro    | 0.999243  |
| Recall Macro       | 0.999535  |
| F1 Macro           | 0.999389  |

## 5. Local-only Training ✅

### Kết quả từng client:

| Client  | Test Accuracy | Test Precision Macro | Test Recall Macro | Test F1 Macro |
| ------- | ------------- | -------------------- | ----------------- | ------------- |
| Client 1 | 0.968070 | 0.956892 | 0.982573 | 0.968070 |
| Client 2 | 0.999316 | 0.998958 | 0.999194 | 0.999076 |
| Client 3 | 0.942023 | 0.923024 | 0.933516 | 0.923380 |
| Client 4 | 0.946702 | 0.973317 | 0.899455 | 0.930684 |
| Client 5 | 0.978876 | 0.970145 | 0.957219 | 0.961592 |

--- 
### Trung bình local (local_avg):
| Metric          | Score     |
| --------------- | --------- |
| Accuracy        | 0.966997  |
| Precision Macro | 0.964467  |
| Recall Macro    | 0.954392  |
| F1 Macro        | 0.956561  |

---

## 6. Federated Learning — Flower ✅

#### FL convergence
| Round | Accuracy  | F1 Macro  |
| ----- | --------- | --------- |
| 0     | 0.165447  | 0.113881  |
| 1     | 0.941050  | 0.924582  |
| 5     | 0.994401  | 0.992602  |
| 10    | 0.998841  | 0.998098  |
| 11    | 0.999651  | 0.999546  |
| 12    | 0.999659  | 0.999550  |

---

#### Kết quả best (Flower)
| Metric             | Score     |
| ------------------ | --------- |
| Accuracy           | 0.999659  |
| Precision Macro    | 0.999473  |
| Recall Macro       | 0.999626  |
| F1 Macro           | 0.999550  |

---

#### Kết luận học thuật
- **FL chênh Centralized < 0.0002 F1** — gần như tương đương (không chia sẻ raw data).
- **FL vượt Local avg +0.043 F1** — chứng minh giá trị của FL trên phân phối dữ liệu Non-IID.


---

## 7. Compare Summary
| setting             | accuracy  | precision_macro | recall_macro | f1_macro  |
| ------------------- | --------- | --------------- | ------------ | --------- |
| flower_best         | 0.999659  | 0.999473        | 0.999626     | 0.999550  |
| centralized         | 0.999536  | 0.999243        | 0.999535     | 0.999389  |
| local_best_client_2 | 0.999316  | 0.998958        | 0.999194     | 0.999076  |
| local_client_2      | 0.999316  | 0.998958        | 0.999194     | 0.999076  |
| local_client_1      | 0.968070  | 0.956892        | 0.982573     | 0.968070  |
| local_client_5      | 0.978876  | 0.970145        | 0.957219     | 0.961592  |
| local_avg           | 0.966997  | 0.964467        | 0.954392     | 0.956561  |
| local_client_4      | 0.946702  | 0.973317        | 0.899455     | 0.930684  |
| local_client_3      | 0.942023  | 0.923024        | 0.933516     | 0.923380  |


Plots đã generate: `experiments/plots/*.png`

---

## 8. FastAPI Inference Backend ✅

### Đã hoàn thành
- `src/inference/api_server.py` — FastAPI app đầy đủ
- `src/inference/predict.py` — `WearableInferenceEngine`
- `src/inference/feature_builder.py` — build feature từ sensor data
- `src/inference/utils.py` — FEATURE_COLS, LABEL_MAP

### Endpoints
| Endpoint | Method | Mô tả |
|---|---|---|
| `/health` | GET | Kiểm tra server + model |
| `/labels` | GET | Map 0→OK, 1→MEDIUM, 2→HIGH |
| `/device-activity-map` | GET | Map device code → activity group |
| `/predict` | POST | Predict từ 8 features trực tiếp |
| `/predict-from-sensor` | POST | Predict từ raw sensor data (Flutter dùng cái này) |

### Request format Flutter gửi lên (`/predict-from-sensor`)
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

### Chạy backend
```bash
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` bắt buộc để điện thoại trong cùng WiFi truy cập được.

### Lưu ý quan trọng
- Model file: `models/flower_global_best.pt` (không phải `fl_global_best.pt` — server.py lưu tên khác với api_server.py load)
- Scaler file: `models/flower_scaler.pkl`
- Cần mở firewall Windows port 8000 nếu dùng điện thoại thật

---

## 9. SQLite Logger ✅

### Đã hoàn thành
- `src/inference/logger.py` — đầy đủ 4 tables + helper functions

### Schema
```sql
CREATE TABLE readings (id, timestamp, client_id, hr_bpm, activity, activity_name,
                       risk_prob_ok, risk_prob_med, risk_prob_high, alert, source);
CREATE TABLE alerts   (id, timestamp, client_id, hr_bpm, severity, resolved, resolved_at);
CREATE TABLE fl_rounds  (round_id, timestamp, n_clients, avg_accuracy, avg_f1_macro);
CREATE TABLE fl_clients (id, round_id, client_id, local_f1, fl_f1, n_samples);
```

### Đã có functions
- `init_db()` — tạo DB và bảng từ schema.sql
- `log_reading()` — log mỗi BPM reading + probs + alert
- `log_alert()` — log khi severity là MEDIUM hoặc HIGH
- `resolve_alert()` — đánh dấu alert đã xử lý
- `log_fl_round()` / `log_fl_client()` — log FL metrics
- `get_recent_readings()`, `get_unresolved_alerts()`, `get_alert_summary()`

### Tích hợp
Logger hiện được gọi trong `realtime_loop.py`. Với kiến trúc Option B, logger nên được tích hợp vào `api_server.py` hoặc chạy song song — cần quyết định:
- [ ] Thêm `log_reading()` call vào endpoint `/predict-from-sensor` của `api_server.py`

---

## 10. Real-time Inference Loop (Serial) ✅

### Trạng thái
`src/inference/realtime_loop.py` đã hoàn chỉnh với model Flower mới:
- Load `flower_global_best.pt` + `flower_scaler.pkl`
- 8 features đúng thứ tự FEATURE_COLS
- Gửi `'A'`/`'M'`/`'O'` về ESP32 qua Serial
- Log vào SQLite qua `logger.py`
- Demo mode (không cần ESP32): `python src/inference/realtime_loop.py`
- Serial mode: `python src/inference/realtime_loop.py --port COM5`

### Lưu ý
Với kiến trúc **Option B** (Flutter làm trung gian), `realtime_loop.py` **không còn là primary flow** cho demo. Nó vẫn giữ lại như:
- Fallback khi không có điện thoại
- Tool để test model độc lập
- Demo mode cho báo cáo

---

## 11. Flutter Mobile App

### 11.1 Đã hoàn thành (cấu trúc)
Toàn bộ cấu trúc app đã có:

```
lib/
├── app.dart, main.dart
├── core/         config.dart, constants.dart, theme.dart, utils.dart
├── models/       app_settings, history_item, predict_response,
│                 sensor_packet, sensor_request
├── providers/    app_state, prediction_provider, sensor_provider,
│                 settings_provider
├── routes/       app_routes.dart
├── screens/      about, alert, device, history, home, live_monitor,
│                 settings, splash
├── services/     api_service, device_service, history_service,
│                 mock_sensor_service, mqtt_service, notification_service,
│                 settings_service
└── widgets/      alert_banner, history_tile, metric_tile, prediction_card,
                  probability_bar, sensor_input_form
```

Screens đã implement: HomeScreen, LiveMonitorScreen, HistoryScreen, SettingsScreen, DeviceScreen, SplashScreen, AboutScreen.

Mock sensor hoạt động: 4 profiles (rest/walk/brisk/run) với noise ngẫu nhiên.

### 11.2 Cần sửa cho Option B (⚠️ phần quan trọng nhất còn lại)

**`lib/services/device_service.dart`** — rewrite hoàn toàn:
- Implement MQTT connect tới `broker.hivemq.com:1883`
- Subscribe `wearable/data` → parse JSON → emit `SensorPacket`
- Method `sendAlertCommand(String cmd)` — publish `wearable/alert`
- Auto-reconnect khi mất kết nối
- Client ID unique: `"flutter_app_${timestamp}"`

**`lib/providers/sensor_provider.dart`** — đổi `deviceService` thành `public`:
```dart
final DeviceService deviceService = DeviceService(); // public
```

**`lib/providers/prediction_provider.dart`** — thêm parameter `deviceService`:
```dart
Future<void> predictFromPacket({
  ...
  DeviceService? deviceService,  // MỚI
}) async {
  // sau khi predict:
  final cmd = result.predClass == 2 ? 'A'
             : result.predClass == 1 ? 'M' : 'O';
  await deviceService?.sendAlertCommand(cmd);
}
```

**`lib/screens/live_monitor_screen.dart`** — thêm auto predict Timer:
```dart
Timer.periodic(Duration(ms: settings.predictIntervalMs), (_) {
  if (settings.autoPredict && packet != null && !prediction.isLoading) {
    prediction.predictFromPacket(..., deviceService: sensor.deviceService);
  }
});
```

C��p nhật tất cả calls `predictFromPacket` truyền thêm `deviceService: sensor.deviceService`.

**`lib/core/config.dart`**:
```dart
static String defaultBaseUrl = 'http://10.0.2.2:8000'; // emulator
// Đổi thành IP LAN khi test trên điện thoại thật
static const int defaultPredictIntervalMs = 2000; // giảm xuống 2s
```

### 11.3 Dependency cần có trong pubspec.yaml
```yaml
mqtt_client: ^10.0.0        # MQTT 2 chiều
fl_chart: ^0.68.0           # Realtime HR chart
flutter_local_notifications  # Push notification
provider: ^6.0.0
shared_preferences           # Lưu settings + history
uuid: ^4.0.0                 # HistoryItem ID
http: ^1.0.0                 # API calls
intl: ^0.18.0                # Date formatting
```

---

## 12. Adafruit IO Dashboard ❌ (chưa bắt đầu)

### Feeds cần tạo
- `heart-rate` — HR gauge real-time
- `activity-level` — rest/walk/brisk/run
- `alert-cmd` — OK/MEDIUM/HIGH
- `fl-accuracy` — line chart FL accuracy per round

### Cần làm
- [ ] Tạo Adafruit IO account + feeds + dashboard widgets
- [ ] Viết `gateway/module1_display.py` — publish lên Adafruit feeds

> Với Option B, ESP32 → MQTT → Flutter → API, Adafruit dashboard có thể bị loại khỏi flow chính. Cân nhắc: có thể publish song song từ ESP32 lên cả `wearable/data` (HiveMQ) lẫn Adafruit MQTT trong cùng `TaskCommunicate`.

---

## 13. Báo cáo cuối kỳ ⚠️ (đang làm — 25%)

### Đã có
- Results & Discussion section hoàn chỉnh
- README đầy đủ để tham chiếu

### Cần hoàn thiện
- [ ] Chương 1: Giới thiệu đề tài
- [ ] Chương 2: Cơ sở lý thuyết (IoT, FL, PAMAP2, MLP, Flutter, MQTT)
- [ ] Chương 3: Thiết kế hệ thống — cập nhật kiến trúc Option B, use-case, ER diagram, UI/UX
- [ ] Chương 4: Hiện thực (ESP32 firmware, Strategy Pattern, SQLite, FL pipeline)
- [ ] Chương 5: Kết quả thực nghiệm ← đã có data
- [ ] Chương 6: Privacy Discussion (FL vs centralized, gradient leakage, GDPR)
- [ ] Chương 7: Kết luận & Hướng phát triển
- [ ] Phụ lục: Code snippets, screenshots

---

## 14. Tóm tắt công việc còn lại (ưu tiên theo demo)

### Ưu tiên 1 — ESP32 firmware Option B (CE-1/CE-2) — ~2 giờ
```
[ ] globals.h: thêm AlertLevel enum
[ ] globals.cpp: init alertLevel = ALERT_OK
[ ] config.h: đổi MQTT_SERVER = "broker.hivemq.com"
[ ] comm_task.cpp: rewrite — bỏ Serial, thêm mqttCallback,
    subscribe wearable/alert, publish wearable/data
[ ] alert_manager.cpp: 3 mức dựa trên alertLevel (switch/case)
[ ] pulse_sensor.cpp: bỏ hardcode test alertTriggered
[ ] Flash + test: Serial Monitor thấy "Subscribed to wearable/alert"
```

### Ưu tiên 2 — Flutter DeviceService MQTT (CE-3/CE-4) — ~3 giờ
```
[ ] device_service.dart: rewrite implement MQTT 2 chiều
[ ] sensor_provider.dart: deviceService public
[ ] prediction_provider.dart: thêm deviceService param, gửi A/M/O
[ ] live_monitor_screen.dart: thêm auto-predict Timer
[ ] Cập nhật tất cả predictFromPacket calls
[ ] config.dart: đổi defaultBaseUrl, predictIntervalMs = 2000
[ ] Test: app nhận data từ ESP32 → predict → LED/Buzzer phản ứng
```

### Ưu tiên 3 — Backend logger integration (CS-1) — ~30 phút
```
[ ] api_server.py: thêm log_reading() call trong /predict-from-sensor
[ ] Kiểm tra SQLite ghi đúng sau mỗi predict
```

### Ưu tiên 4 — Firewall + network setup (tất cả) — ~15 phút
```
[ ] Mở Windows Firewall port 8000
[ ] Xác nhận IP laptop: ipconfig
[ ] Test từ điện thoại: browser → http://<IP>:8000/health
[ ] Settings app: cập nhật Base URL đúng IP
```

### Ưu tiên 5 — Adafruit IO Dashboard (CE-4) — tuỳ thời gian
```
[ ] Tạo account + feeds
[ ] module1_display.py
```

### Ưu tiên 6 — Báo cáo (cả team) — song song
```
[ ] Viết 7 chương, target ≥ 30 trang
[ ] Cập nhật sơ đồ kiến trúc theo Option B
```

---

## 15. Checklist demo end-to-end

```
Chuẩn bị (T-15 phút):
  □ ESP32 flash firmware mới, cắm nguồn
  □ Serial Monitor: kiểm tra "Subscribed to wearable/alert"
  □ Laptop: uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000
  □ Laptop: GET /health → {"status": "ok"}
  □ Điện thoại: Settings → Base URL = http://<IP laptop>:8000
  □ Điện thoại: "Check Backend" → "Backend healthy"
  □ Điện thoại: Mode = real_device

Khi demo:
  □ Live Monitor → Connect Device (MQTT kết nối HiveMQ)
  □ Bật Auto Predict (interval 2s)
  □ Quan sát: HR chart cập nhật từ ESP32 thật
  □ Quan sát: PredictionCard chuyển màu OK/MEDIUM/HIGH
  □ Quan sát: ESP32 LED nhấp nháy / Buzzer kêu đúng mức
  □ History tab: readings được lưu

Fallback nếu MQTT/WiFi không ổn:
  □ Mode = mock → chọn profile (rest/walk/brisk/run)
  □ Demo model predict bình thường, không có LED/Buzzer thật
  □ Hoặc: python src/inference/realtime_loop.py (Serial mode)
```

---

## 16. Files quan trọng — trạng thái hiện tại

```
fl_wearable/
├── hardware/src/
│   ├── main.cpp              ✅
│   ├── pulse_sensor.cpp      ⚠️ cần bỏ hardcode test
│   ├── motion_sensor.cpp     ✅
│   ├── comm_task.cpp         ⚠️ cần rewrite cho Option B
│   ├── alert_manager.cpp     ⚠️ cần 3-level switch/case
│   ├── config.h              ⚠️ cần đổi MQTT_SERVER
│   └── globals.h/cpp         ⚠️ cần AlertLevel enum
│
├── src/
│   ├── data/                 ✅ tất cả
│   ├── models/mlp.py         ✅
│   ├── training/             ✅ tất cả
│   ├── fl/                   ✅ tất cả
│   └── inference/
│       ├── api_server.py     ✅ (cần thêm log_reading call)
│       ├── predict.py        ✅
│       ├── feature_builder.py ✅
│       ├── logger.py         ✅
│       ├── realtime_loop.py  ✅ (fallback, không phải primary)
│       └── utils.py          ✅
│
├── models/
│   ├── flower_global_best.pt ✅
│   ├── flower_scaler.pkl     ✅
│   ├── centralized_mlp.pt    ✅
│   └── local_client_1..5.pt  ✅
│
├── lib/ (Flutter)
│   ├── core/                 ✅ (config.dart cần đổi URL)
│   ├── models/               ✅ tất cả
│   ├── providers/
│   │   ├── app_state.dart         ✅
│   │   ├── settings_provider.dart ✅
│   │   ├── sensor_provider.dart   ⚠️ cần deviceService public
│   │   └── prediction_provider.dart ⚠️ cần thêm deviceService param
│   ├── screens/              ✅ (live_monitor cần auto-predict Timer)
│   ├── services/
│   │   ├── api_service.dart        ✅
│   │   ├── device_service.dart     ⚠️ cần rewrite MQTT 2 chiều
│   │   ├── mock_sensor_service.dart ✅
│   │   ├── mqtt_service.dart       ✅ (base, device_service dùng pattern này)
│   │   ├── history_service.dart    ✅
│   │   ├── notification_service.dart ✅
│   │   └── settings_service.dart   ✅
│   └── widgets/              ✅ tất cả
│
├── database/
│   └── schema.sql            ✅ (được tạo bởi logger.py khi init_db())
│
├── experiments/
│   ├── compare_summary.csv   ✅
│   ├── fl_round_metrics.csv  ✅
│   └── plots/*.png           ✅
│
├── README.md                 ✅
└── requirements.txt          ✅
```

---

*Checkpoint cập nhật 12/05/2026. Kiến trúc đã chốt Option B — Flutter làm trung gian. CS pipeline (data → FL → inference API) hoàn chỉnh 100%. Phần còn lại tập trung vào: (1) rewrite ESP32 comm_task cho MQTT 2 chiều, (2) rewrite Flutter DeviceService, (3) wire lại prediction flow gửi alert về hardware.*