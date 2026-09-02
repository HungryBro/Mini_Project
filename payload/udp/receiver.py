"""Receive and display traffic.v1 UDP payloads for a local demo."""

from __future__ import annotations

import argparse
import json
import socket
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive traffic.v1 JSON payloads over UDP")
    parser.add_argument("--host", default="127.0.0.1", help="local address to bind")
    parser.add_argument("--port", type=int, default=5005, help="UDP port to bind")
    parser.add_argument("--once", action="store_true", help="stop after the first valid payload")
    return parser.parse_args()


def print_payload(payload: dict[str, Any], address: tuple[str, int], byte_count: int) -> None:
    location = payload.get("location", {})
    traffic = payload.get("traffic", {})
    wrong_way = payload.get("wrong_way", {})
    print(
        f"[UDP IN] from {address[0]}:{address[1]} | {byte_count} bytes | "
        f"{location.get('site_id', 'unknown')} / {location.get('camera_id', 'unknown')} | "
        f"vehicles={traffic.get('unique_vehicle_count', 0)} wrong_way={wrong_way.get('count', 0)}"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    received = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.host, args.port))
        print(f"Traffic UDP receiver listening on udp://{args.host}:{args.port} (Ctrl+C to stop)")
        try:
            while True:
                data, address = sock.recvfrom(65_535)
                try:
                    payload = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    print(f"[IGNORED] invalid JSON from {address[0]}:{address[1]}: {error}")
                    continue
                print_payload(payload, address, len(data))
                received += 1
                if args.once:
                    break
        except KeyboardInterrupt:
            pass
    print(f"Receiver stopped after {received} valid payload(s)")


if __name__ == "__main__":
    main()
