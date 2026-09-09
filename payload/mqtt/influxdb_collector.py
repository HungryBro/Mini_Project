"""Write Krung Thon Bridge V2 MQTT summaries directly into InfluxDB.

This service runs on the Mac. It subscribes to the Cloud VerneMQ broker and
writes each summary to the Cloud InfluxDB API, so Docker, Kafka, and a Cloud
Telegraf installation are not required for the storage path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PAYLOAD_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PAYLOAD_DIRECTORY.parent
if str(PAYLOAD_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_DIRECTORY))

import paho.mqtt.client as mqtt

from mqtt.settings import MQTT_BROKER, MQTT_PORT, MQTT_QOS, STUDENT_ID, SUMMARY_TOPIC


DEFAULT_ENV_PATH = (
    PROJECT_DIRECTORY / "infrastructure" / "traffic-cloud-telegraf" / ".env"
)
MEASUREMENT = f"traffic_{STUDENT_ID}"
COLLECTOR_CLIENT_ID = f"influx_collector_{STUDENT_ID}"

TAG_PATHS = {
    "field_id": "field_id",
    "name": "name",
    "place_id": "place_id",
    "lane_1_direction": "payload.lane_1_direction",
    "lane_2_direction": "payload.lane_2_direction",
    "lane_3_direction": "payload.lane_3_direction",
    "lane_4_direction": "payload.lane_4_direction",
}
FIELD_TYPES = {
    "vehicle_count": "int",
    "wrong_way_count": "int",
    "wrong_way_rate_per_100_vehicles": "float",
    "lane_1_direction_value": "int",
    "lane_2_direction_value": "int",
    "lane_3_direction_value": "int",
    "lane_4_direction_value": "int",
    "lane_1_vehicle_count": "int",
    "lane_1_wrong_way_count": "int",
    "lane_1_wrong_way_rate_per_100": "float",
    "lane_2_vehicle_count": "int",
    "lane_2_wrong_way_count": "int",
    "lane_2_wrong_way_rate_per_100": "float",
    "lane_3_vehicle_count": "int",
    "lane_3_wrong_way_count": "int",
    "lane_3_wrong_way_rate_per_100": "float",
    "lane_4_vehicle_count": "int",
    "lane_4_wrong_way_count": "int",
    "lane_4_wrong_way_rate_per_100": "float",
}


def load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Collector environment file not found: {path}")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    required = ("INFLUX_URL", "INFLUX_ORG", "INFLUX_BUCKET", "INFLUX_TOKEN")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError("Missing required .env keys: " + ", ".join(missing))
    return values


def nested_value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"MQTT payload is missing {path}")
        value = value[key]
    return value


def escape_measurement(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ")


def escape_tag(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(" ", "\\ ")
        .replace(",", "\\,")
        .replace("=", "\\=")
    )


def format_field(value: Any, field_type: str) -> str:
    if field_type == "bool":
        return "true" if bool(value) else "false"
    if field_type == "int":
        return f"{int(value)}i"

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Influx field must be finite, got {value!r}")
    return repr(converted)


def to_line_protocol(payload: dict[str, Any]) -> str:
    """Map the Option A V2 summary to one Influx point."""
    field_id = str(nested_value(payload, "field_id"))
    if field_id != STUDENT_ID:
        raise ValueError(f"Unexpected field_id {field_id!r}; expected {STUDENT_ID!r}")

    tags = ",".join(
        f"{name}={escape_tag(nested_value(payload, path))}"
        for name, path in TAG_PATHS.items()
    )
    fields = ",".join(
        f"{name}={format_field(nested_value(payload, f'payload.{name}'), field_type)}"
        for name, field_type in FIELD_TYPES.items()
    )
    timestamp = int(nested_value(payload, "payload.timestamp"))
    if timestamp <= 0:
        raise ValueError("payload.timestamp must be a positive Unix timestamp")
    return f"{escape_measurement(MEASUREMENT)},{tags} {fields} {timestamp}"


def write_to_influx(line_protocol: str, env: dict[str, str]) -> None:
    query = urlencode(
        {
            "org": env["INFLUX_ORG"],
            "bucket": env["INFLUX_BUCKET"],
            "precision": "s",
        }
    )
    request = Request(
        f"{env['INFLUX_URL'].rstrip('/')}/api/v2/write?{query}",
        data=line_protocol.encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Token {env['INFLUX_TOKEN']}",
            "Content-Type": "text/plain; charset=utf-8",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=15) as response:
                if response.status != 204:
                    raise RuntimeError(f"InfluxDB returned HTTP {response.status}")
            return
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(f"InfluxDB write failed after 3 attempts: {last_error}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to V2 traffic MQTT summaries and write them to InfluxDB."
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to the InfluxDB/MQTT environment file.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after one successfully written live MQTT summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    env = load_env(args.env)
    completed = False

    try:
        client = mqtt.Client(
            client_id=COLLECTOR_CLIENT_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except AttributeError:
        client = mqtt.Client(client_id=COLLECTOR_CLIENT_ID)

    def on_connect(
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        if reason_code == 0:
            client.subscribe(SUMMARY_TOPIC, qos=MQTT_QOS)
            print(
                f"Collector connected to {MQTT_BROKER}:{MQTT_PORT}; "
                f"subscribed to {SUMMARY_TOPIC}",
                flush=True,
            )
        else:
            print(f"[MQTT ERROR] Connection rejected: {reason_code}", flush=True)

    def on_message(client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        nonlocal completed
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("MQTT body is not a JSON object")
            line_protocol = to_line_protocol(payload)
            write_to_influx(line_protocol, env)
            values = payload["payload"]
            print(
                f"[INFLUX WRITE OK] measurement={MEASUREMENT} "
                f"field_id={STUDENT_ID} timestamp={values['timestamp_th']} "
                f"vehicles={values['vehicle_count']} "
                f"wrong_way={values['wrong_way_count']}",
                flush=True,
            )
            completed = True
            if args.once:
                client.disconnect()
        except Exception as error:
            print(f"[COLLECTOR ERROR] {error}", flush=True)

    client.on_connect = on_connect
    client.on_message = on_message

    print(
        f"Starting direct MQTT-to-Influx collector for {MEASUREMENT}; "
        f"bucket={env['INFLUX_BUCKET']}",
        flush=True,
    )
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nCollector stopped.", flush=True)
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
    if args.once and not completed:
        raise SystemExit("Collector stopped before a summary was written.")


if __name__ == "__main__":
    main()
