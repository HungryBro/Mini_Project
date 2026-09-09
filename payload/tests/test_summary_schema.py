"""Checks for the MQTT summary schema shared by Gateway, Influx, and Kafka."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mqtt.gateway import to_option_a_payload
from mqtt.influxdb_collector import to_line_protocol


class SummarySchemaTests(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "id": "ID_6610301004",
            "student_id": "6610301004",
            "timestamp": "2026-09-09T15:12:40+07:00",
            "timestamp_unix": 1788941560,
            "location": {"camera_id": "CAM_112", "site_id": "krung_thon_bridge"},
            "traffic": {
                "vehicle_count": 92,
                "lane_vehicle_counts": {f"lane_{number}": number for number in range(1, 5)},
            },
            "wrong_way": {
                "count": 1,
                "rate_per_100_vehicles": 1.09,
                "by_lane": {
                    f"lane_{number}": {
                        "count": 1 if number == 2 else 0,
                        "rate_per_100_vehicles": 3.45 if number == 2 else 0.0,
                    }
                    for number in range(1, 5)
                },
            },
            "lanes": {
                "lane_1": {"direction": "up"},
                "lane_2": {"direction": "up"},
                "lane_3": {"direction": "down"},
                "lane_4": {"direction": "down"},
            },
        }

    def test_gateway_places_timestamps_at_top_level_only(self):
        result = to_option_a_payload(self.summary)
        self.assertEqual(result["timestamp"], 1788941560)
        self.assertEqual(result["timestamp_th"], "2026-09-09T15:12:40+07:00")
        self.assertNotIn("timestamp", result["payload"])
        self.assertNotIn("timestamp_th", result["payload"])
        self.assertNotIn("name", result)
        self.assertNotIn("wrong_way_rate_per_100_vehicles", result["payload"])
        self.assertFalse(any("rate_per_100" in key for key in result["payload"]))

    def test_influx_uses_top_level_timestamp_as_point_time(self):
        result = to_option_a_payload(self.summary)
        line = to_line_protocol(result)
        self.assertTrue(line.endswith(" 1788941560"))
        self.assertNotIn("timestamp=", line)


if __name__ == "__main__":
    unittest.main()
