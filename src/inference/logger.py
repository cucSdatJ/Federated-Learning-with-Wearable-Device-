import sqlite3
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path("database/fl_wearable.db")
SCHEMA_PATH = Path("database/schema.sql")

ACTIVITY_NAMES = {
    0: "rest",
    1: "walk",
    2: "brisk",
    4: "run",
}


# ==========================================
# INIT
# ==========================================
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    print(f"[DB] Initialized: {db_path.resolve()}")
    return conn


# ==========================================
# LOG READING
# ==========================================
def log_reading(
    conn: sqlite3.Connection,
    hr_bpm: float,
    activity_code: int,
    probs: list,          # [prob_ok, prob_med, prob_high]
    alert: str,           # 'OK' | 'MEDIUM' | 'HIGH'
    client_id: str = "esp32_01",
    source: str = "realtime_loop",
):
    timestamp = datetime.now().isoformat()
    activity_name = ACTIVITY_NAMES.get(activity_code, "unknown")

    conn.execute(
        """
        INSERT INTO readings
            (timestamp, client_id, hr_bpm, activity, activity_name,
             risk_prob_ok, risk_prob_med, risk_prob_high, alert, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp, client_id, hr_bpm, activity_code, activity_name,
            float(probs[0]), float(probs[1]), float(probs[2]),
            alert, source,
        ),
    )
    conn.commit()


# ==========================================
# LOG ALERT
# ==========================================
def log_alert(
    conn: sqlite3.Connection,
    hr_bpm: float,
    severity: str,        # 'MEDIUM' | 'HIGH'
    client_id: str = "esp32_01",
):
    timestamp = datetime.now().isoformat()

    conn.execute(
        """
        INSERT INTO alerts (timestamp, client_id, hr_bpm, severity)
        VALUES (?, ?, ?, ?)
        """,
        (timestamp, client_id, hr_bpm, severity),
    )
    conn.commit()


# ==========================================
# RESOLVE ALERT
# ==========================================
def resolve_alert(conn: sqlite3.Connection, alert_id: int):
    resolved_at = datetime.now().isoformat()
    conn.execute(
        "UPDATE alerts SET resolved=1, resolved_at=? WHERE id=?",
        (resolved_at, alert_id),
    )
    conn.commit()


# ==========================================
# LOG FL ROUND
# ==========================================
def log_fl_round(
    conn: sqlite3.Connection,
    round_id: int,
    n_clients: int,
    avg_accuracy: float,
    avg_f1_macro: float,
):
    timestamp = datetime.now().isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO fl_rounds
            (round_id, timestamp, n_clients, avg_accuracy, avg_f1_macro)
        VALUES (?, ?, ?, ?, ?)
        """,
        (round_id, timestamp, n_clients, avg_accuracy, avg_f1_macro),
    )
    conn.commit()


# ==========================================
# LOG FL CLIENT
# ==========================================
def log_fl_client(
    conn: sqlite3.Connection,
    round_id: int,
    client_id: int,
    local_f1: float,
    fl_f1: float,
    n_samples: int,
):
    conn.execute(
        """
        INSERT INTO fl_clients
            (round_id, client_id, local_f1, fl_f1, n_samples)
        VALUES (?, ?, ?, ?, ?)
        """,
        (round_id, client_id, local_f1, fl_f1, n_samples),
    )
    conn.commit()


# ==========================================
# QUERY HELPERS
# ==========================================
def get_recent_readings(conn: sqlite3.Connection, limit: int = 20):
    cursor = conn.execute(
        "SELECT * FROM readings ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    return cursor.fetchall()


def get_unresolved_alerts(conn: sqlite3.Connection):
    cursor = conn.execute(
        "SELECT * FROM alerts WHERE resolved=0 ORDER BY timestamp DESC"
    )
    return cursor.fetchall()


def get_alert_summary(conn: sqlite3.Connection):
    cursor = conn.execute(
        """
        SELECT
            severity,
            COUNT(*) as total,
            SUM(resolved) as resolved_count
        FROM alerts
        GROUP BY severity
        """
    )
    return cursor.fetchall()


# ==========================================
# TEST / DEMO
# ==========================================
if __name__ == "__main__":
    conn = init_db()

    # Test log_reading
    log_reading(conn, hr_bpm=80.0,  activity_code=0, probs=[1.0, 0.0, 0.0], alert="OK")
    log_reading(conn, hr_bpm=102.0, activity_code=0, probs=[0.001, 0.999, 0.0], alert="MEDIUM")
    log_reading(conn, hr_bpm=175.0, activity_code=4, probs=[0.0, 0.0, 1.0], alert="HIGH")

    # Test log_alert
    log_alert(conn, hr_bpm=102.0, severity="MEDIUM")
    log_alert(conn, hr_bpm=175.0, severity="HIGH")

    # Test log_fl_round
    log_fl_round(conn, round_id=15, n_clients=5, avg_accuracy=0.9985, avg_f1_macro=0.9977)
    log_fl_client(conn, round_id=15, client_id=1, local_f1=0.91, fl_f1=0.998, n_samples=248599)

    # Query
    print("\n[RECENT READINGS]")
    for row in get_recent_readings(conn, limit=5):
        print(f"  {row['timestamp']} | {row['hr_bpm']} bpm | {row['activity_name']} | {row['alert']}")

    print("\n[UNRESOLVED ALERTS]")
    for row in get_unresolved_alerts(conn):
        print(f"  {row['timestamp']} | {row['hr_bpm']} bpm | {row['severity']}")

    print("\n[ALERT SUMMARY]")
    for row in get_alert_summary(conn):
        print(f"  {row['severity']}: total={row['total']} resolved={row['resolved_count']}")

    conn.close()
    print("\n[DONE]")