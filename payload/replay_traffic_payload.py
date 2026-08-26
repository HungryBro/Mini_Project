"""Replay a vehicle_tracking JSONL file as compact UDP traffic payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any

from traffic_payload import TrafficWindowAggregator, parse_start_time


DEFAULT_INPUT = Path("runs/krung_thon_bridge/v2/v2_wrong_way_coco/krung_thon_bridge_cam112_v2_1min_tracks.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay tracked traffic JSONL as one UDP payload per time window")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="vehicle_tracking JSONL input")
    parser.add_argument("--host", default="127.0.0.1", help="UDP receiver host")
    parser.add_argument("--port", type=int, default=5005, help="UDP receiver port")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="source-time aggregation window")
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
    parser.add_argument("--dry-run", action="store_true", help="print payloads without sending UDP")
    return parser.parse_args()


def send_payload(sock: socket.socket, payload: dict[str, Any], host: str, port: int, dry_run: bool) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    location = payload["location"]
    summary = payload["traffic"]
    wrong_way = payload["wrong_way"]
    action = "DRY RUN" if dry_run else "UDP OUT"
    print(
        f"[{action}] {location['camera_id']} {payload['window']['start']} -> {payload['window']['end']} "
        f"| vehicles={summary['unique_vehicle_count']} wrong_way={wrong_way['count']} bytes={len(encoded)}"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not dry_run:
        sock.sendto(encoded, (host, port))


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input JSONL not found: {args.input}")
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be greater than zero")
    if args.send_delay < 0:
        raise SystemExit("--send-delay cannot be negative")

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
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at {args.input}:{line_number}: {error.msg}") from error
            frames += 1
            for payload in aggregator.add_frame(record):
                send_payload(sock, payload, args.host, args.port, args.dry_run)
                sent += 1
                if args.send_delay:
                    time.sleep(args.send_delay)

        final_payload = aggregator.flush()
        if final_payload:
            send_payload(sock, final_payload, args.host, args.port, args.dry_run)
            sent += 1

    print(f"Replay complete: {frames} frames -> {sent} traffic payload(s)")


if __name__ == "__main__":
    main()
