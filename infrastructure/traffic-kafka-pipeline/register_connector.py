"""Create or update the MQTT-to-Kafka source connector for traffic summaries."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


BASE_DIRECTORY = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def setting(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def connector_config() -> tuple[str, str, dict[str, str]]:
    name = "traffic-mqtt-source"
    connect_url = setting("CONNECT_URL", "http://localhost:8083").rstrip("/")
    mqtt_broker = setting("MQTT_BROKER", "172.16.2.117")
    mqtt_port = setting("MQTT_PORT", "1883")
    config = {
        "connector.class": "io.confluent.connect.mqtt.MqttSourceConnector",
        "tasks.max": "1",
        "mqtt.server.uri": f"tcp://{mqtt_broker}:{mqtt_port}",
        "mqtt.topics": setting(
            "MQTT_TOPIC", "v1/6610301004"
        ),
        "mqtt.client.id": setting(
            "MQTT_CONNECT_CLIENT_ID", "kafka_connect_traffic_CAM_112"
        ),
        "mqtt.qos": setting("MQTT_QOS", "1"),
        "mqtt.clean.session": "false",
        "kafka.topic": setting(
            "KAFKA_TOPIC", "traffic.krung_thon_bridge.summary.v1"
        ),
        "key.converter": "org.apache.kafka.connect.storage.StringConverter",
        "value.converter": "org.apache.kafka.connect.converters.ByteArrayConverter",
        "confluent.topic.bootstrap.servers": "kafka:29092",
        "confluent.topic.replication.factor": "1",
    }
    return name, connect_url, config


def request_json(url: str, *, method: str = "GET", payload: Any = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        content = response.read().decode("utf-8")
    return json.loads(content) if content else None


def wait_for_connect(connect_url: str, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            request_json(f"{connect_url}/connector-plugins")
            return
        except (HTTPError, URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Kafka Connect was not ready at {connect_url} within {wait_seconds:g}s"
                )
            time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register the VerneMQ-to-Kafka traffic connector"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the connector JSON only"
    )
    parser.add_argument(
        "--wait-seconds", type=float, default=180.0, help="Kafka Connect startup wait"
    )
    args = parser.parse_args()

    load_dotenv(BASE_DIRECTORY / ".env")
    name, connect_url, config = connector_config()

    if args.dry_run:
        print(json.dumps({"name": name, "config": config}, indent=2))
        return

    try:
        print(f"Waiting for Kafka Connect at {connect_url}...")
        wait_for_connect(connect_url, args.wait_seconds)
        config_url = f"{connect_url}/connectors/{quote(name, safe='')}/config"
        request_json(config_url, method="PUT", payload=config)
        status = request_json(f"{connect_url}/connectors/{quote(name, safe='')}/status")
    except (HTTPError, URLError, TimeoutError) as error:
        raise SystemExit(f"Connector registration failed: {error}") from error

    print("MQTT -> Kafka connector is registered.")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
