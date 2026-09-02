"""Separate MQTT Gateway for the Krung Thon Bridge V2 Tracker.

Subscribes to the Tracker's one-minute gateway-input topic, prints the complete
payload, combines five minutes of traffic, then publishes one cloud summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PAYLOAD_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PAYLOAD_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_DIRECTORY))

import paho.mqtt.client as mqtt

from mqtt.settings import (
    GATEWAY_CLIENT_ID,
    GATEWAY_CLOUD_WINDOW_SECONDS,
    GATEWAY_INPUT_TOPIC,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_QOS,
    SUMMARY_TOPIC,
)
from common.traffic_payload import TrafficGatewayAggregator


def print_payload(label: str, payload: dict[str, Any], *, topic: str) -> None:
    window = payload.get("window", {})
    traffic = payload.get("traffic", {})
    wrong_way = payload.get("wrong_way", {})
    print(
        f"[{label}] topic={topic} "
        f"{window.get('start')} -> {window.get('end')} "
        f"| vehicles={traffic.get('vehicle_count', 0)} "
        f"wrong_way={wrong_way.get('count', 0)}",
        flush=True,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    aggregator = TrafficGatewayAggregator(window_seconds=GATEWAY_CLOUD_WINDOW_SECONDS)

    try:
        client = mqtt.Client(
            client_id=GATEWAY_CLIENT_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except AttributeError:
        client = mqtt.Client(client_id=GATEWAY_CLIENT_ID)

    def publish_summary(payload: dict[str, Any]) -> None:
        print_payload("GATEWAY MQTT OUT", payload, topic=SUMMARY_TOPIC)
        result = client.publish(SUMMARY_TOPIC, json.dumps(payload, ensure_ascii=False), qos=MQTT_QOS)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT ERROR] Summary publish failed: rc={result.rc}", flush=True)

    def on_connect(client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        # paho-mqtt 2.x passes a ReasonCode object here.  It compares safely
        # with 0, while int(ReasonCode) raises TypeError on recent releases.
        if reason_code == 0:
            client.subscribe(GATEWAY_INPUT_TOPIC, qos=MQTT_QOS)
            print("==================================================")
            print(" MQTT Gateway started")
            print(f" Input topic : {GATEWAY_INPUT_TOPIC} (every 1 minute)")
            print(f" Output topic: {SUMMARY_TOPIC} (every 5 minutes)")
            print(" Press Ctrl+C to stop")
            print("==================================================", flush=True)
        else:
            print(f"[MQTT ERROR] Gateway connection rejected: {reason_code}", flush=True)

    def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            print(f"[IGNORED] Invalid gateway JSON on {msg.topic}: {error}", flush=True)
            return
        if not isinstance(payload, dict):
            print(f"[IGNORED] Gateway payload on {msg.topic} is not a JSON object", flush=True)
            return
        print_payload("GATEWAY MQTT IN", payload, topic=msg.topic)
        for cloud_payload in aggregator.add_payload(payload):
            publish_summary(cloud_payload)

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print(f"Connecting MQTT Gateway to {MQTT_BROKER}:{MQTT_PORT}...", flush=True)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping MQTT Gateway...", flush=True)
    except Exception as error:
        print(f"[MQTT ERROR] Gateway could not run: {error}", flush=True)
    finally:
        final_payload = aggregator.flush(complete_window=False)
        if final_payload:
            publish_summary(final_payload)
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
