"""Replay a vehicle_tracking JSONL file as compact UDP traffic payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any

PAYLOAD_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PAYLOAD_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_DIRECTORY))

from common.traffic_payload import TrafficWindowAggregator, parse_start_time


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents)
        if (candidate / "runs" / "krung_thon_bridge").is_dir()
    ),
    SCRIPT_DIR,
)
DEFAULT_INPUT = PROJECT_ROOT / "runs/krung_thon_bridge/v2/v2_wrong_way_coco/krung_thon_bridge_cam112_v2_1min_tracks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay tracked traffic JSONL as one UDP/MQTT payload per time window")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="vehicle_tracking JSONL input")
    parser.add_argument("--host", default="127.0.0.1", help="UDP receiver host")
    parser.add_argument("--port", type=int, default=5005, help="UDP receiver port")
    parser.add_argument("--window-seconds", type=float, default=30.0, help="source-time aggregation window")
    parser.add_argument("--site-id", default="krung_thon_bridge")
    parser.add_argument("--site-name", default="Krung Thon Bridge")
    parser.add_argument("--camera-id", default="CAM_112")
    parser.add_argument("--camera-name", default="Main traffic camera")
    parser.add_argument(
        "--start-time",
        default=None,
        help="ISO-8601 time for source second 0; default is the current UTC time",
    )
    parser.add_argument("--send-delay", type=float, default=0.0, help="seconds to wait after each sent summary")
    parser.add_argument("--dry-run", action="store_true", help="print payloads without sending UDP/MQTT")
    parser.add_argument("--mqtt", action="store_true", help="publish payloads over MQTT as well")
    parser.add_argument("--mqtt-broker", default="broker.hivemq.com", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--mqtt-topic", default="traffic/krung_thon_bridge/summary", help="MQTT topic")
    return parser.parse_args()


def send_payload(
    sock: socket.socket,
    mqtt_client: Any | None,
    payload: dict[str, Any],
    host: str,
    port: int,
    mqtt_topic: str,
    dry_run: bool,
) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    location = payload["location"]
    summary = payload["traffic"]
    wrong_way = payload["wrong_way"]
    action = "DRY RUN" if dry_run else "PAYLOAD OUT"
    site_label = location.get("camera_id", location.get("site_id", "site"))
    print(
        f"[{action}] {site_label} {payload['window']['start']} -> {payload['window']['end']} "
        f"| vehicles={summary['unique_vehicle_count']} wrong_way={wrong_way['count']} bytes={len(encoded)}"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not dry_run:
        sock.sendto(encoded, (host, port))
        if mqtt_client:
            try:
                mqtt_client.publish(mqtt_topic, encoded)
                print(f"[MQTT OUT] Published to {mqtt_topic}")
            except Exception as err:
                print(f"[MQTT ERROR] Could not publish: {err}")


def find_latest_live_log() -> Path | None:
    live_logs_dir = PROJECT_ROOT / "runs/live_logs"
    if not live_logs_dir.is_dir():
        return None
    logs = sorted(
        live_logs_dir.glob("**/*_camera112.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def main() -> None:
    args = parse_args()
    input_path = args.input
    latest_live = find_latest_live_log()

    # If --input was not explicitly specified or points to default, prioritize latest live_logs
    if args.input == DEFAULT_INPUT and latest_live:
        input_path = latest_live

    if not input_path.is_absolute() and not input_path.is_file():
        project_relative = PROJECT_ROOT / input_path
        if project_relative.is_file():
            input_path = project_relative
    if not input_path.is_file() and latest_live:
        input_path = latest_live
    if not input_path.is_file():
        raise SystemExit(f"Input JSONL not found: {input_path}")
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be greater than zero")
    if args.send_delay < 0:
        raise SystemExit("--send-delay cannot be negative")

    mqtt_client = None
    if args.mqtt and not args.dry_run:
        try:
            import paho.mqtt.client as mqtt
            try:
                mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError:
                mqtt_client = mqtt.Client()
            mqtt_client.connect(args.mqtt_broker, args.mqtt_port, 60)
            mqtt_client.loop_start()
            print(f"Connected to MQTT broker {args.mqtt_broker}:{args.mqtt_port}")
        except Exception as err:
            print(f"[MQTT WARNING] Could not connect to MQTT broker: {err}")
            mqtt_client = None

    aggregator = TrafficWindowAggregator(
        window_seconds=args.window_seconds,
        camera_id=args.camera_id,
        site_id=args.site_id,
        site_name=args.site_name,
        camera_name=args.camera_name,
        anchor_time=parse_start_time(args.start_time),
    )
    frames = 0
    sent = 0
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"Invalid JSON at {input_path}:{line_number}: {error.msg}") from error
                frames += 1
                for payload in aggregator.add_frame(record):
                    send_payload(sock, mqtt_client, payload, args.host, args.port, args.mqtt_topic, args.dry_run)
                    sent += 1
                    if args.send_delay:
                        time.sleep(args.send_delay)

            final_payload = aggregator.flush()
            if final_payload:
                send_payload(sock, mqtt_client, final_payload, args.host, args.port, args.mqtt_topic, args.dry_run)
                sent += 1
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()

    print(f"Replay complete: {frames} frames -> {sent} traffic payload(s)")


if __name__ == "__main__":
    main()
