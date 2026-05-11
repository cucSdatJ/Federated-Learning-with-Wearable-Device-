-- FL Wearable Database Schema

CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    client_id   TEXT    NOT NULL DEFAULT 'esp32_01',
    hr_bpm      REAL    NOT NULL,
    activity    INTEGER NOT NULL,
    activity_name TEXT  NOT NULL,
    risk_prob_ok    REAL NOT NULL DEFAULT 0.0,
    risk_prob_med   REAL NOT NULL DEFAULT 0.0,
    risk_prob_high  REAL NOT NULL DEFAULT 0.0,
    alert       TEXT    NOT NULL CHECK(alert IN ('OK','MEDIUM','HIGH')),
    source      TEXT    NOT NULL DEFAULT 'realtime_loop'
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    client_id   TEXT    NOT NULL DEFAULT 'esp32_01',
    hr_bpm      REAL    NOT NULL,
    severity    TEXT    NOT NULL CHECK(severity IN ('MEDIUM','HIGH')),
    resolved    INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS fl_rounds (
    round_id    INTEGER PRIMARY KEY,
    timestamp   TEXT    NOT NULL,
    n_clients   INTEGER NOT NULL,
    avg_accuracy  REAL  NOT NULL,
    avg_f1_macro  REAL  NOT NULL
);

CREATE TABLE IF NOT EXISTS fl_clients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id    INTEGER NOT NULL,
    client_id   INTEGER NOT NULL,
    local_f1    REAL    NOT NULL,
    fl_f1       REAL    NOT NULL,
    n_samples   INTEGER NOT NULL,
    FOREIGN KEY (round_id) REFERENCES fl_rounds(round_id)
);

CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp   ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved    ON alerts(resolved);