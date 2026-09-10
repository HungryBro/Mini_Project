"""MQTT settings shared by the V2 Tracker, Gateway, and monitor."""

MQTT_BROKER = "172.16.2.117"
MQTT_PORT = 1883
MQTT_QOS = 1

# Keep this as text so the student identifier is never reformatted as a number.
STUDENT_ID = "6610301004"

# Tracker publishes one local window each minute to this topic.
GATEWAY_INPUT_TOPIC = "traffic/krung_thon_bridge/CAM_112/gateway_input"
# Publish the V2 summary in the legacy-compatible topic format required by
# the existing lab MQTT-to-Kafka connector.  This ``v1`` is a topic protocol,
# not the archived V1 tracker.
SUMMARY_TOPIC = f"v1/{STUDENT_ID}"

TRACKER_CLIENT_ID = "vehicle_tracker_CAM_112"
GATEWAY_CLIENT_ID = f"vehicle_gateway_{STUDENT_ID}_CAM_112"
GATEWAY_CLOUD_WINDOW_SECONDS = 60.0

# Cloud Kafka; no local Docker broker is needed.
KAFKA_BOOTSTRAP_SERVERS = "172.16.2.117:9092"
KAFKA_TOPIC = "traffic_6610301004"
KAFKA_BRIDGE_CLIENT_ID = f"mqtt_kafka_bridge_{STUDENT_ID}"
KAFKA_ML_CONSUMER_GROUP = f"traffic_ml_{STUDENT_ID}"
# The Lab advertises this alias. Translate it only inside our Kafka client.
# If the Lab later advertises its Cloud IP directly, this entry is simply unused.
KAFKA_ADDRESS_MAP = {"localhost:9092": "172.16.2.117:9092"}
