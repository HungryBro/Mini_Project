"""MQTT settings shared by the V2 Tracker, Gateway, and monitor."""

MQTT_BROKER = "172.16.2.117"
MQTT_PORT = 1883
MQTT_QOS = 1

# Keep this as text so the student identifier is never reformatted as a number.
STUDENT_ID = "6610301004"

# Tracker publishes one local window each minute to this topic.
GATEWAY_INPUT_TOPIC = "traffic/krung_thon_bridge/CAM_112/gateway_input"
# Option A: publish the legacy-compatible envelope that the existing lab
# pipeline accepts through its ``v1/+`` subscription.
SUMMARY_TOPIC = f"v1/{STUDENT_ID}"

TRACKER_CLIENT_ID = "vehicle_tracker_CAM_112"
GATEWAY_CLIENT_ID = "vehicle_gateway_CAM_112"
GATEWAY_CLOUD_WINDOW_SECONDS = 60.0
