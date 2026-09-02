"""Receive and display traffic.v1 JSON payloads over MQTT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PAYLOAD_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PAYLOAD_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_DIRECTORY))

from mqtt.settings import MQTT_BROKER, MQTT_PORT, SUMMARY_TOPIC

try:
    import paho.mqtt.client as mqtt
except ImportError:
    raise SystemExit("paho-mqtt library is required. Install it using: pip install paho-mqtt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive traffic.v1 JSON payloads over MQTT")
    parser.add_argument("--broker", default=MQTT_BROKER, help="MQTT broker host")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help="MQTT broker port")
    parser.add_argument(
        "--topic", default=SUMMARY_TOPIC, help="MQTT topic to subscribe"
    )
    parser.add_argument("--once", action="store_true", help="stop after the first valid payload")
    return parser.parse_args()


def print_payload(payload: dict[str, Any], topic: str, byte_count: int, broker: str) -> None:
    legacy_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else None
    location = payload.get("location", {}) if legacy_payload is None else {"site_id": payload.get("place_id")}
    traffic = payload.get("traffic", {}) if legacy_payload is None else legacy_payload
    wrong_way = payload.get("wrong_way", {}) if legacy_payload is None else legacy_payload
    wrong_way_count = (
        wrong_way.get("count", 0)
        if legacy_payload is None
        else wrong_way.get("wrong_way_count", 0)
    )
    print(
        f"[MQTT IN] broker={broker} | topic={topic} | {byte_count} bytes | "
        f"{location.get('site_id', 'unknown')} | "
        f"vehicles={traffic.get('vehicle_count', 0)} wrong_way={wrong_way_count}"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    received_count = 0

    def on_connect(client: mqtt.Client, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        if rc == 0:
            print(f"Connected to MQTT broker {args.broker}:{args.port}")
            client.subscribe(args.topic)
            print(f"Subscribed to topic: {args.topic} (Ctrl+C to stop)")
        else:
            print(f"Failed to connect to MQTT broker, return code {rc}")

    def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        nonlocal received_count
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            print(f"[IGNORED] invalid JSON on topic {msg.topic}: {error}")
            return

        print_payload(payload, msg.topic, len(msg.payload), args.broker)
        received_count += 1
        if args.once:
            client.disconnect()

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker, args.port, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(f"MQTT Receiver error: {err}")
    finally:
        client.disconnect()

    print(f"MQTT Receiver stopped after {received_count} valid payload(s)")


if __name__ == "__main__":
    main()
