"""Kafka Cloud -> Worker A on Mac -> Influx Cloud. See README.md."""
from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "payload"))
from ml.lane_analysis import MODEL_VERSION, analyze, observed_fields, parse_summary
from mqtt.influxdb_collector import DEFAULT_ENV_PATH, escape_tag, format_field, load_env, write_to_influx
from mqtt.kafka_connection import connection_config
from mqtt.settings import KAFKA_ADDRESS_MAP, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, STUDENT_ID

DB_PATH = ROOT / "runs" / "ml_a" / "history.sqlite3"
GROUP = f"traffic_lane_effect_{STUDENT_ID}_v1"


def open_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript('''
        CREATE TABLE IF NOT EXISTS observations (
            place TEXT, ts INTEGER, body TEXT NOT NULL,
            PRIMARY KEY(place, ts));
        CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY, line TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rejected (
            topic TEXT, partition_id INTEGER, offset_id INTEGER, reason TEXT, raw BLOB,
            PRIMARY KEY(topic, partition_id, offset_id));
        CREATE TABLE IF NOT EXISTS fits (place TEXT PRIMARY KEY, ts INTEGER NOT NULL);
    ''')
    return db


def point(measurement, row, fields, status=None):
    tags = f"field_id={STUDENT_ID},place_id={escape_tag(row['place_id'])},model_version={MODEL_VERSION}"
    if status:
        tags += f",status={status}"
    else:
        tags += f",lane_mode={'3_1' if row['mode'] else '2_2'}"
    values = ",".join(f"{k}={format_field(v, 'int' if type(v) is int else 'float')}" for k, v in fields.items())
    return f"{measurement},{tags} {values} {row['timestamp']}"


def stage(db, row):
    """Persist history and deterministic output together before Kafka acknowledgment."""
    place, ts = row["place_id"], row["timestamp"]
    encoded = json.dumps(row, sort_keys=True)
    existing = db.execute("SELECT body FROM observations WHERE place=? AND ts=?", (place, ts)).fetchone()
    if existing:
        if existing[0] != encoded:
            raise ValueError("Conflicting summaries at identical place/timestamp")
        return "duplicate"
    with db:
        db.execute("INSERT INTO observations VALUES (?,?,?)", (place, ts, encoded))
        db.execute("INSERT INTO outbox(line) VALUES (?)", (
            point(f"traffic_lane_observed_{STUDENT_ID}", row, observed_fields(row)),))
        last = db.execute("SELECT ts FROM fits WHERE place=?", (place,)).fetchone()
        if last and ts < last[0] + 600:
            return "observed"
        history = [json.loads(r[0]) for r in db.execute(
            "SELECT body FROM observations WHERE place=? AND ts BETWEEN ? AND ? ORDER BY ts",
            (place, ts - 30 * 86400, ts))]
        status, fields = analyze(history)
        db.execute("INSERT INTO outbox(line) VALUES (?)", (
            point(f"traffic_lane_model_{STUDENT_ID}", row, fields, status),))
        db.execute("INSERT OR REPLACE INTO fits VALUES (?,?)", (place, ts))
        return status


def flush(db, env):
    pending = db.execute("SELECT id,line FROM outbox ORDER BY id LIMIT 100").fetchall()
    if not pending:
        return 0
    write_to_influx("\n".join(r[1] for r in pending), env)
    with db:
        db.executemany("DELETE FROM outbox WHERE id=?", [(r[0],) for r in pending])
    print(f"[ML A INFLUX OK] points={len(pending)} bucket={env['INFLUX_BUCKET']}", flush=True)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--once", action="store_true", help="Stop after one valid Kafka record and pending writes")
    parser.add_argument("--timeout", type=int, default=120, help="Idle timeout for --once")
    parser.add_argument("--status", action="store_true", help="Read local history/outbox without network")
    parser.add_argument("--offline", type=Path, help="Analyze JSONL locally; never connect to Kafka/Influx or production DB")
    args = parser.parse_args()
    if args.offline:
        rows = [parse_summary(line) for line in args.offline.read_bytes().splitlines() if line.strip()]
        unique = {}
        for row in rows:
            key = (row['place_id'], row['timestamp'])
            if key in unique and unique[key] != row:
                raise ValueError("Conflicting summaries at identical place/timestamp")
            unique[key] = row
        for place in sorted({r['place_id'] for r in unique.values()}):
            subset = [r for r in unique.values() if r['place_id'] == place]
            latest = max(r['timestamp'] for r in subset)
            status, fields = analyze([r for r in subset if r['timestamp'] >= latest - 30 * 86400])
            print(json.dumps(dict(place_id=place, status=status, **fields), indent=2))
        return
    if args.status:
        if not args.db.exists():
            print("No local Worker A history yet")
            return
        db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        for table in ("observations", "outbox", "rejected"):
            print(f"{table}={db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]}")
        print("latest_source_timestamp=", db.execute("SELECT MAX(ts) FROM observations").fetchone()[0])
        db.close()
        return

    from kafka import KafkaConsumer, TopicPartition
    from kafka.structs import OffsetAndMetadata
    env = load_env(args.env)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    lock = args.db.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = open_db(args.db)
    config = connection_config(KAFKA_BOOTSTRAP_SERVERS, KAFKA_ADDRESS_MAP, GROUP)
    config["request_timeout_ms"] = 40000
    consumer = KafkaConsumer(KAFKA_TOPIC, group_id=GROUP, enable_auto_commit=False,
                             auto_offset_reset="earliest", max_poll_records=1, **config)
    print(f"[ML A START] kafka={KAFKA_BOOTSTRAP_SERVERS} topic={KAFKA_TOPIC} group={GROUP}", flush=True)
    print("[ML A] lane 3; rolling 30 days; minimum 14 days and 60 nonzero windows per mode", flush=True)
    deadline = time.monotonic() + args.timeout
    processed = False
    try:
        while True:
            try:
                flush(db, env)
            except RuntimeError as error:
                print(f"[ML A RETRY] {error}; retained in SQLite", flush=True)
                # Exit rather than lose Kafka ownership in an unbounded network retry.
                raise
            if args.once and processed and not db.execute("SELECT 1 FROM outbox LIMIT 1").fetchone():
                return
            batch = consumer.poll(timeout_ms=1000, max_records=1)
            for records in batch.values():
                for record in records:
                    try:
                        row = parse_summary(record.value)
                        status = stage(db, row)
                        print(f"[ML A RECORD] timestamp={row['timestamp']} mode={row['mode']} status={status}", flush=True)
                        processed = True
                    except (ValueError, KeyError, TypeError, OverflowError) as error:
                        with db:
                            db.execute("INSERT OR REPLACE INTO rejected VALUES (?,?,?,?,?)",
                                       (record.topic, record.partition, record.offset, str(error), record.value))
                        print(f"[ML A REJECTED] offset={record.offset}: {error}", flush=True)
                    # Commit ONLY this record, after durable local history/outbox or rejection.
                    consumer.commit({TopicPartition(record.topic, record.partition):
                                     OffsetAndMetadata(record.offset + 1, "", -1)})
            if args.once and not processed and time.monotonic() >= deadline:
                raise RuntimeError("No valid Kafka record before timeout; check tracker, gateway and bridge")
    finally:
        consumer.close(autocommit=False)
        db.close()
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ML A STOPPED] Local history/outbox retained")
    except Exception as error:
        print(f"[ML A STOPPED] {error}", file=sys.stderr)
        sys.exit(1)
