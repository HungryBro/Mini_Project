"""Consume traffic summaries from Kafka, score anomalies, and write InfluxDB."""
from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer
from sklearn.ensemble import IsolationForest

PAYLOAD_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PAYLOAD_DIRECTORY.parent
if str(PAYLOAD_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_DIRECTORY))

from mqtt.influxdb_collector import (
    DEFAULT_ENV_PATH,
    escape_measurement,
    escape_tag,
    format_field,
    load_env,
    write_to_influx,
)
from mqtt.kafka_connection import connection_config
from mqtt.settings import (
    KAFKA_ADDRESS_MAP,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_ML_CONSUMER_GROUP,
    KAFKA_TOPIC,
    STUDENT_ID,
)


MEASUREMENT = f"traffic_ml_{STUDENT_ID}"
MODEL_NAME = "isolation_forest"
MODEL_RANDOM_STATE = 301004
DEFAULT_MIN_TRAINING_SAMPLES = 10
DEFAULT_HISTORY_SIZE = 500

# Every feature is numeric and already present in the current MQTT/Kafka schema.
FEATURE_NAMES = (
    "vehicle_count",
    "wrong_way_count",
    "lane_1_direction_value",
    "lane_2_direction_value",
    "lane_3_direction_value",
    "lane_4_direction_value",
    "lane_1_vehicle_count",
    "lane_1_wrong_way_count",
    "lane_2_vehicle_count",
    "lane_2_wrong_way_count",
    "lane_3_vehicle_count",
    "lane_3_wrong_way_count",
    "lane_4_vehicle_count",
    "lane_4_wrong_way_count",
)


def parse_summary(raw: bytes) -> dict[str, Any]:
    body = json.loads(raw)
    if not isinstance(body, dict) or str(body.get("field_id")) != STUDENT_ID:
        raise ValueError(f"Kafka record must belong to field_id={STUDENT_ID}")
    if type(body.get("timestamp")) is not int or body["timestamp"] <= 0:
        raise ValueError("Kafka record needs a positive integer timestamp")
    if not isinstance(body.get("timestamp_th"), str) or not body["timestamp_th"]:
        raise ValueError("Kafka record needs timestamp_th")
    if not isinstance(body.get("place_id"), str) or not body["place_id"]:
        raise ValueError("Kafka record needs place_id")
    payload = body.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Kafka record needs a payload object")

    features = []
    for name in FEATURE_NAMES:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"payload.{name} must be numeric")
        features.append(float(value))

    return_id = f"{STUDENT_ID}:{body['place_id']}:{body['timestamp']}"
    return {
        "message_id": message_id,
        "field_id": STUDENT_ID,
        "place_id": body["place_id"],
        "timestamp": body["timestamp"],
        "timestamp_th": body["timestamp_th"],
        "features": features,
        "vehicle_count": int(payload["vehicle_count"]),
        "wrong_way_count": int(payload["wrong_way_count"]),
    }


def train_and_score(history: list[list[float]], current: list[float]) -> tuple[int, float]:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(history)
    anomaly_flag = 1 if int(model.predict([current])[0]) == -1 else 0
    # sklearn's decision_function is lower for anomalies; invert it so a
    # larger dashboard value consistently means more anomalous.
    anomaly_score = -float(model.decision_function([current])[0])
    return anomaly_flag, anomaly_score


def ml_line_protocol(summary: dict[str, Any], flag: int, score: float, training_samples: int) -> str:
    tags = ",".join(
        (
            f"field_id={escape_tag(summary['field_id'])}",
            f"place_id={escape_tag(summary['place_id'])}",
            f"model={escape_tag(MODEL_NAME)}",
        )
    )
    fields = ",".join(
        (
            f"anomaly_flag={format_field(flag, 'int')}",
            f"anomaly_score={format_field(score, 'float')}",
            f"training_samples={format_field(training_samples, 'int')}",
            f"vehicle_count={format_field(summary['vehicle_count'], 'int')}",
            f"wrong_way_count={format_field(summary['wrong_way_count'], 'int')}",
        )
    )
    return f"{escape_measurement(MEASUREMENT)},{tags} {fields} {summary['timestamp']}"


def open_state(path: Path) -> sqlite3.Connection:
    database = sqlite3.connect(path)
    database.execute(
        """CREATE TABLE IF NOT EXISTS samples (
               message_id TEXT PRIMARY KEY,
               timestamp INTEGER NOT NULL,
               features TEXT NOT NULL,
               ml_written INTEGER NOT NULL CHECK (ml_written IN (0, 1))
           )"""
    )
    database.commit()
    return database


def history(database: sqlite3.Connection, limit: int) -> list[list[float]]:
    rows = database.execute(
        "SELECT features FROM samples ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return [json.loads(row[0]) for row in reversed(rows)]


def remember(database: sqlite3.Connection, summary: dict[str, Any], ml_written: bool) -> None:
    database.execute(
        "INSERT OR IGNORE INTO samples(message_id, timestamp, features, ml_written) VALUES (?, ?, ?, ?)",
        (
            summary["message_id"],
            summary["timestamp"],
            json.dumps(summary["features"], separators=(",", ":")),
            int(ml_written),
        ),
    )
    database.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--min-training", type=int, default=DEFAULT_MIN_TRAINING_SAMPLES)
    parser.add_argument("--history-size", type=int, default=DEFAULT_HISTORY_SIZE)
    parser.add_argument("--once", action="store_true", help="Stop after one ML result is written.")
    args = parser.parse_args()
    if args.min_training < 2:
        parser.error("--min-training must be at least 2")
    if args.history_size < args.min_training:
        parser.error("--history-size must be at least --min-training")
    return args


def main() -> None:
    args = parse_args()
    env = load_env(args.env_file)
    state_directory = PROJECT_DIRECTORY / "runs" / "ml_worker"
    state_directory.mkdir(parents=True, exist_ok=True)
    lock = (state_directory / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another traffic ML worker is already running.")

    database = open_state(state_directory / "state.sqlite3")
    consumer = KafkaConsumer(
        args.topic,
        **connection_config(args.bootstrap, KAFKA_ADDRESS_MAP, "traffic_ml_" + STUDENT_ID),
        group_id=KAFKA_ML_CONSUMER_GROUP,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_records=1,
    )
    written = 0
    print(
        f"ML worker started: Kafka {args.bootstrap} topic={args.topic} -> "
        f"InfluxDB measurement={MEASUREMENT}",
        flush=True,
    )
    try:
        while not (args.once and written):
            records = consumer.poll(timeout_ms=1000, max_records=1)
            for batch in records.values():
                for record in batch:
                    try:
                        summary = parse_summary(record.value)
                    except (ValueError, json.JSONDecodeError) as error:
                        print(f"[ML IGNORED] partition={record.partition} offset={record.offset}: {error}", flush=True)
                        consumer.commit()
                        continue

                    if database.execute(
                        "SELECT 1 FROM samples WHERE message_id=?", (summary["message_id"],)
                    ).fetchone():
                        print(f"[ML DUPLICATE] {summary['message_id']}", flush=True)
                        consumer.commit()
                        continue

                    training = history(database, args.history_size)
                    if len(training) < args.min_training:
                        remember(database, summary, ml_written=False)
                        consumer.commit()
                        print(
                            f"[ML WARMUP] samples={len(training) + 1}/{args.min_training} "
                            f"timestamp={summary['timestamp_id']}",
                            flush=True,
                        )
                        continue

                    flag, score = train_and_score(training, summary["features"])
                    line = ml_line_protocol(summary, flag, score, len(training))
                    write_to_influx(line, env)
                    remember(database, summary, ml_written=True)
                    consumer.commit()
                    written += 1
                    print(
                        f"[ML WRITE OK] measurement={MEASUREMENT} timestamp={summary['timestamp_th']} "
                        f"anomaly_flag={flag} anomaly_score={score:.6f} "
                        f"training_samples={len(training)}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("ML worker stopped.", flush=True)
    finally:
        consumer.close()
        database.close()
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ML WORKER STOPPED] {error}", file=sys.stderr, flush=True)
        sys.exit(1)
