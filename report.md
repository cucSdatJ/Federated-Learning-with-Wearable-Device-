# Final Report Outline  
**Federated Learning for Personalized Wearable Device Control**  
**Course**: Information Systems  
**Mandatory modules**: 4  
**Main goal**: ≥ 30 pages of content  

---

## 1. Introduction(*)

### 1.1 Problem Statement and Research Motivation
- The need for health monitoring via wearable devices in the IoT era.
- Limitations of centralized models: risk of exposing sensitive physiological data.
- Federated Learning as a privacy-preserving solution.

### 1.2 Project Goals and Scope
- Classify heart rate risk levels (OK / MEDIUM / HIGH) according to activity context.
- Train FL on PAMAP2 dataset without sharing raw data between clients.
- Build an end-to-end system with 4 modules, 4-layer architecture.  

**Out-of-scope**: clinical validation, Differential Privacy.

### 1.3 Hardware Description
- **ESP32 Wearable**: Pulse Sensor (pin 32), MPU6050 IMU (I2C), Buzzer (pin 27), Red LED (pin 2).
- **Laptop/PC Gateway**: runs Python inference + FL training (not on MCU).
- **FreeRTOS**: 4 parallel tasks on core 0 + core 1.

### 1.4 Mandatory 4 Modules
1. ESP32 Firmware — sensor reading, Serial/MQTT communication.
2. Python Gateway — real-time inference, FL training (Flower/FedAvg).
3. FastAPI Backend — REST API + SQLite logging.
4. Flutter Mobile App — display HR, alerts, MQTT subscription.

### 1.5 Overall 4-Layer Architecture
- Layer 1: Hardware — ESP32 + sensors.
- Layer 2: Gateway — Python laptop (inference center required by spec).
- Layer 3: Cloud/Broker — Core IoT.
- Layer 4: Mobile — Flutter App display layer.

### 1.6 Group Roles
- CS member: data pipeline, ML/FL, FastAPI, Flutter app, real-time inference.
- CE member: ESP32 firmware (FreeRTOS, sensors, MQTT).

---

## 2. Theoretical Background(*)

### 2.1 Internet of Things and MQTT
- IoT architecture: Perception → Network → Application.
- MQTT protocol: Pub/Sub model, QoS, Broker.
- Brokers used: broker.hivemq.com:1883 (Flutter↔ESP32), app.coreiot.io (Core IoT telemetry).
- Core IoT platform and Access Token authentication.

### 2.2 Embedded Programming: ESP32 and FreeRTOS
- Real-Time Operating System: task scheduling, priority, core affinity.
- `xTaskCreatePinnedToCore`: pin task to specific core.
- Shared variables with `volatile`.
- Arduino/C++ vs MicroPython — reasoning for choice.

### 2.3 Federated Learning
- Definition, privacy issues in traditional ML.
- FL architecture: Server, Clients, Round, Aggregation.
- FedAvg algorithm (McMahan et al., 2017): weighted averaging by samples.
- Non-IID data: heterogeneity among wearable clients.
- Flower framework: WearableFlowerClient, NumPyClient, FedAvg strategy.
- Reason for 1 epoch/round: prevent client drift on Non-IID data.

### 2.4 MLP and PyTorch
- Multi-layer Perceptron: Linear layers, activation functions.
- BatchNormalization, Dropout: regularization.
- CrossEntropyLoss with class weights: handling class imbalance.
- Adam optimizer, StandardScaler.

### 2.5 PAMAP2 Dataset
- 9 subjects, 13 activity IDs.
- IMU 100Hz + HR 4Hz.
- NaN handling: linear interpolation for HR.
- Natural Non-IID features.

### 2.6 FastAPI, SQLite and Flutter
- FastAPI: async REST API, Pydantic schema.
- SQLite: embedded DB.
- Flutter: cross-platform, mqtt_client, charts, notifications.

---

## 3. System Design

### 3.1 Detailed Architecture
- 4-layer diagram with full data flow:
  - ESP32 → Serial → Python Gateway → MQTT → Core IoT → Flutter
  - Python Gateway → REST API (FastAPI) → SQLite

### 3.2 MQTT Topic Architecture
- wearable/data: ESP32 publishes {bpm, activity}
- wearable/alert: Flutter subscribes
- Core IoT telemetry
- Serial protocol mapping.

### 3.3 Database Design (SQLite)
- ER diagram: 4 tables.
- `readings`, `alerts`, `fl_rounds`, `fl_clients`.

### 3.4 Use-Case Diagram
- Actors: User, Admin.
- UC-01: Real-time HR monitoring.
- UC-02: Receive HIGH risk alert.
- UC-03: View HR history.
- UC-04: Trigger FL training.
- UC-05: Query data via API.
- ..

### 3.5 Feature Engineering(*)
- 8 features sourced from sensors:
  - Pulse Sensor features (HR mean/std)
  - IMU magnitude
  - One-hot activity encoding

### 3.6 Pseudo-labeling(*)
- Activity grouping by HR distribution.
- Thresholds per group.
- Classes: OK/MEDIUM/HIGH.

### 3.7 FL System(*)
- Topology: 9 clients + 1 server.
- FedAvg weighted aggregation.
- 1 epoch/round.
- 15 rounds

### 3.8 UI/UX Flutter App(*)
- Screens: Home, Alerts, History, Settings,..

---

## 4. System Implementation(*)

### 4.1 ESP32 Firmware
- PlatformIO project structure.
- Tasks for reading pulse, motion, communication, alerts.
- Managing shared globals.

### 4.2 Python Gateway: Inference
- Real-time loop reading sensors.
- Feature builder.
- Model/scaler loading.
- Softmax prediction and alerts.

### 4.3 Python Gateway: FL Training
- Data preprocessing pipeline.
- `split_clients.py`: create 9 client CSVs.
- Flower client/server scripts.
- Baseline training modes: centralized, local-only.

### 4.4 FastAPI Backend
- Predict endpoint.
- Load model/scaler.
- Logging to SQLite.

### 4.5 Flutter Mobile App
- MQTT service.
- UI for HR gauge, activity, alert risks.
- Push notifications.

### 4.6 End-to-End Integration
- WiFi connectivity issues.
- Model/scaler pairing.
- MQTT client ID uniqueness.
- Bug fixes.

---

## 5. Experimental Results

### 5.1 Experimental Environment
- Hardware, dataset statistics.
- Software stack.

### 5.2 Data Pipeline Results
- Activity distribution charts.
- Non-IID label distribution.

### 5.3 FL Convergence (15 Rounds)
- Round metrics table & curves.

### 5.4 Comparison of Configurations
- Centralized baseline.
- FL best.
- Local-only averages.

### 5.5 Result Analysis
- High results explained by pseudo-label circularity.
- FL close to centralized upper bound.
- Collaborative learning improves generalization.

### 5.6 Real-time Inference Demo
- Terminal output.
- Adafruit IO and Flutter screenshots.
- Latency measurements.

---

## 6. Privacy Discussion(*)

### 6.1 Sensitive Data Risks
- HR, activity patterns.
- GDPR considerations.

### 6.2 How FL Preserves Privacy
- Raw data stays local.
- Only weights transmitted.
- Comparison to centralized.

### 6.3 Non-IID and FL Design
- Label distribution differences.
- Epoch/round selection rationale.
- Weighted FedAvg.

### 6.4 Privacy Limitations
- Model inversion attacks.
- Lack of secure aggregation.
- Lack of differential privacy.

### 6.5 Improving Privacy
- Secure Aggregation.
- Differential Privacy (Gaussian noise).
- Homomorphic Encryption.

### 6.6 Pseudo-label Methodology Notes
- Circular dependency.
- Non-clinical ground-truth.

---

## 7. Conclusion and Future Work(*)

### 7.1 Conclusion
- End-to-end system: 5 modules, 4-layer architecture.
- FL achieved F1=0.9977, <0.002 lower than centralized.
- FL outperformed local-only by +0.068 F1.

### 7.2 Evaluation Against Task List
- Input feature definition.
- Device architecture.
- Classification model build.
- Local-only evaluation.
- FL training implementation.
- Comparison of local vs FL models.
- Threshold definition.
- Alerts triggering.
- Real-time on-device inference.
- Accuracy + privacy discussion.

### 7.3 Limitations
- Pseudo-label circularity.
- Hardcoded time features.
- Lack of secure aggregation/differential privacy.
- Offline dataset only.

### 7.4 Future Work
- Export model for mobile inference.
- Personalized FL fine-tuning.
- Add DP, secure aggregation.
- Multi-device FL deployment.
- Clinical labels.
- Extend sensors.

---

## References & Appendices

**References:**
- McMahan et al. (2017) — FedAvg.
- Reiss & Stricker (2012) — PAMAP2.
- Beutel et al. (2020) — Flower.
- MQTT Specification.
- ESP32 docs.
- FastAPI, PyTorch, sklearn, Flutter docs.

**Appendices:**
- Database schema.
- Key code snippets.
- System screenshots.
- FL metrics table.
- Per-class reports.

Estimated contents: ~50 pages (excluding cover, TOC, figures, appendices).