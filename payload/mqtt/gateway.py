"""Separate MQTT Gateway for the Krung Thon Bridge V2 Tracker.

Subscribes to the Tracker's 15-second gateway-input topic, prints the complete
payload, combines one minute of traffic, then publishes one cloud summary.
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
    STUDENT_ID,
    SUMMARY_TOPIC,
)
from common.traffic_payload import TrafficGatewayAggregator


def to_option_a_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten the traffic summary into the legacy ``id/payload`` envelope."""
    location = summary.get("location") or {}
    traffic = summary.get("traffic") or {}
    wrong_way = summary.get("wrong_way") or {}
    lanes = summary.get("lanes") or {}
    lane_counts = traffic.get("lane_vehicle_counts") or {}
    wrong_way_by_lane = wrong_way.get("by_lane") or {}

    payload: dict[str, Any] = {
        "timestamp": int(summary.get("timestamp_unix", 0)),
        "timestamp_th": summary.get("timestamp", ""),
        "window_seconds": float((summary.get("window") or {}).get("seconds", 0.0)),
        "complete_window": bool((summary.get("window") or {}).get("complete_window", False)),
        "vehicle_count": int(traffic.get("vehicle_count", 0)),
        "wrong_way_count": int(wrong_way.get("count", 0)),
        "wrong_way_rate_per_100_vehicles": float(
            wrong_way.get("rate_per_100_vehicles", 0.0)
        ),
    }
    for number in range(1, 5):
        lane_id = f"lane_{number}"
        lane = lanes.get(lane_id) or {}
        wrong_way_lane = wrong_way_by_lane.get(lane_id) or {}
        vehicle_count = int(
            lane_counts.get(lane_id, wrong_way_lane.get("vehicle_count", 0))
        )
        wrong_way_count = int(wrong_way_lane.get("count", 0))
        payload.update(
            {
                f"{lane_id}_direction": str(lane.get("direction", "unknown")),
                f"{lane_id}_vehicle_count": vehicle_count,
                f"{lane_id}_wrong_way_count": wrong_way_count,
                f"{lane_id}_wrong_way_rate_per_100": float(
                    wrong_way_lane.get("rate_per_100_vehicles", 0.0)
                ),
            }
        )

    return {
        "id": summary.get("id", f"ID_{summary.get('student_id', '')}"),
        "name": location.get("camera_id", "CAM_112"),
        "place_id": location.get("site_id", "krung_thon_bridge"),
        "payload": payload,
    }


def print_payload(label: str, payload: dict[str, Any], *, topic: str) -> None:
    legacy = payload.get("payload") if isinstance(payload.get("payload"), dict) else None
    window = payload.get("window", {}) if legacy is None else {}
    traffic = payload.get("traffic", {}) if legacy is None else legacy
    wrong_way = payload.get("wrong_way", {}) if legacy is None else legacy
    start = window.get("start") if legacy is None else "legacy"
    end = window.get("end") if legacy is None else legacy.get("timestamp_th")
    wrong_way_count = (
        wrong_way.get("count", 0)
        if legacy is None
        else wrong_way.get("wrong_way_count", 0)
    )
    print(
        f"[{label}] topic={topic} "
        f"{start} -> {end} "
        f"| vehicles={traffic.get('vehicle_count', 0)} "
        f"wrong_way={wrong_way_count}",
        flush=True,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    aggregator = TrafficGatewayAggregator(
        window_seconds=GATEWAY_CLOUD_WINDOW_SECONDS,
        student_id=STUDENT_ID,
    )

    try:
        client = mqtt.Client(
            client_id=GATEWAY_CLIENT_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except AttributeError:
        client = mqtt.Client(client_id=GATEWAY_CLIENT_ID)

    def publish_summary(payload: dict[str, Any]) -> None:
        option_a_payload = to_option_a_payload(payload)
        print_payload("GATEWAY MQTT OUT", option_a_payload, topic=SUMMARY_TOPIC)
        result = client.publish(
            SUMMARY_TOPIC, json.dumps(option_a_payload, ensure_ascii=False), qos=MQTT_QOS
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT ERROR] Summary publish failed: rc={result.rc}", flush=True)

    def on_connect(client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        # paho-mqtt 2.x passes a ReasonCode object here.  It compares safely
        # with 0, while int(ReasonCode) raises TypeError on recent releases.
        if reason_code == 0:
            client.subscribe(GATEWAY_INPUT_TOPIC, qos=MQTT_QOS)
            print("==================================================")
            print(" MQTT Gateway started")
            print(f" Input topic : {GATEWAY_INPUT_TOPIC} (every 15 seconds)")
            print(f" Output topic: {SUMMARY_TOPIC} (every 1 minute)")
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
