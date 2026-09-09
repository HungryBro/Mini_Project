"""Regression checks for the Lab's advertised-address compatibility adapter."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kafka.protocol.metadata import MetadataResponse
from mqtt.kafka_connection import client_factory
from mqtt.kafka_bridge import validate_message


class CloudAddressTests(unittest.TestCase):
    def setUp(self):
        self.client = client_factory(
            '172.16.2.117:9092', {'localhost:9092': '172.16.2.117:9092'},
        )(bootstrap_servers='172.16.2.117:9092', api_version=(2, 6))
        self.addCleanup(self.client.close)

    def update_brokers(self, brokers):
        self.client.cluster.update_metadata(MetadataResponse[1](brokers, 1, []))

    def test_mapping_survives_metadata_refresh(self):
        for _ in range(3):
            self.update_brokers([(1, 'localhost', 9092, None)])
            broker = self.client.cluster.broker_metadata(1)
            self.assertEqual((broker.host, broker.port), ('172.16.2.117', 9092))
            self.assertEqual(next(iter(self.client.cluster.brokers())).host, 'localhost')

    def test_direct_cloud_metadata_needs_no_mapping(self):
        self.update_brokers([(1, '172.16.2.117', 9092, None)])
        self.assertEqual(self.client.cluster.broker_metadata(1).host, '172.16.2.117')

    def test_unknown_local_port_is_not_redirected(self):
        self.update_brokers([(1, 'localhost', 9093, None)])
        with self.assertRaisesRegex(RuntimeError, 'Unmapped'):
            self.client.cluster.broker_metadata(1)

    def test_other_cloud_broker_keeps_its_endpoint(self):
        self.update_brokers([(2, '172.16.2.118', 9093, None)])
        broker = self.client.cluster.broker_metadata(2)
        self.assertEqual((broker.host, broker.port), ('172.16.2.118', 9093))

    def test_mapping_cannot_collapse_two_brokers(self):
        self.update_brokers([(1, 'localhost', 9092, None), (2, '172.16.2.117', 9092, None)])
        with self.assertRaisesRegex(RuntimeError, 'Multiple broker IDs'):
            self.client.cluster.broker_metadata(1)

    def test_mapping_cannot_silently_send_to_another_destination(self):
        with self.assertRaisesRegex(ValueError, 'bootstrap_servers'):
            client_factory('172.16.2.118:9092', {'localhost:9092': '172.16.2.117:9092'})

    def test_mapping_does_not_modify_another_client(self):
        other = client_factory('172.16.2.117:9092', {})(
            bootstrap_servers='172.16.2.117:9092', api_version=(2, 6),
        )
        self.addCleanup(other.close)
        other.cluster.update_metadata(MetadataResponse[1]([(1, 'localhost', 9092, None)], 1, []))
        with self.assertRaisesRegex(RuntimeError, 'Unmapped'):
            other.cluster.broker_metadata(1)


class PayloadTests(unittest.TestCase):
    def test_preserves_original_bytes_and_types(self):
        raw = b'{ "field_id": "6610301004", "timestamp":1788941200, "timestamp_th":"2026-09-09T15:12:40+07:00", "payload": {"vehicle_count":12,"rate":0.0,"lane_1_direction":"up","lane_1_direction_value":1} }'
        self.assertIs(validate_message(raw), raw)

    def test_rejects_another_student(self):
        with self.assertRaises(ValueError):
            validate_message(b'{"field_id":"someone_else","timestamp":1788941200,"payload":{}}')

    def test_rejects_old_nested_timestamp_shape(self):
        with self.assertRaisesRegex(ValueError, 'Missing integer timestamp'):
            validate_message(b'{"field_id":"6610301004","payload":{"timestamp":1788941200}}')

    def test_requires_top_level_timestamp_th(self):
        with self.assertRaisesRegex(ValueError, 'Missing timestamp_th'):
            validate_message(b'{"field_id":"6610301004","timestamp":1788941200,"payload":{}}')


if __name__ == '__main__':
    unittest.main()
