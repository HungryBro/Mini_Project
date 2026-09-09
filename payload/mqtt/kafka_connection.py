"""Kafka connections with an explicit, client-only Lab address mapping.

The Lab's single broker advertises localhost:9092 to external clients. Use the
KafkaProducer/Consumer kafka_client extension point to translate that endpoint
on every broker lookup, including after metadata refresh. No system DNS, hosts
file, listening proxy, or Kafka server configuration is changed.

This adapter is tested with kafka-python 2.2.15 (pinned in requirements.txt).
"""
from __future__ import annotations

import ipaddress

from kafka.client_async import KafkaClient
from kafka.conn import get_ip_port_afi


def endpoint(value):
    host, port, _ = get_ip_port_afi(value)
    return host.lower(), port


def is_local(host):
    if host.lower().rstrip('.') == 'localhost':
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or address.is_unspecified
    except ValueError:
        return False


def client_factory(bootstrap, address_map):
    """Create clients whose explicit aliases resolve to a configured bootstrap.

    Only the configured host AND port are translated. Other Cloud brokers keep
    their advertised endpoints. Unmapped loopback endpoints are rejected.
    """
    allowed = {endpoint(item.strip()) for item in bootstrap.split(',')}
    mapping = {endpoint(source): endpoint(target) for source, target in address_map.items()}
    if any(is_local(host) for host, _ in allowed):
        raise ValueError('Cloud bootstrap must use a reachable Cloud IP/DNS.')
    for target in mapping.values():
        if target not in allowed:
            raise ValueError('Each Kafka address-map target must be in bootstrap_servers.')

    class CloudKafkaClient(KafkaClient):
        def __init__(self, **configs):
            super().__init__(**configs)
            original_lookup = self.cluster.broker_metadata

            def mapped_lookup(node_id):
                broker = original_lookup(node_id)
                if broker is None:
                    return None
                source = (broker.host.lower(), broker.port)
                target = mapping.get(source)
                if target:
                    # Mapping one endpoint cannot represent two different brokers.
                    collisions = [
                        item for item in self.cluster.brokers()
                        if mapping.get((item.host.lower(), item.port),
                                       (item.host.lower(), item.port)) == target
                    ]
                    if len(collisions) > 1:
                        raise RuntimeError('Multiple broker IDs share a mapped endpoint; use distinct Cloud endpoints.')
                    return broker._replace(host=target[0], port=target[1])
                if is_local(broker.host):
                    raise RuntimeError(f'Unmapped Cloud broker endpoint: {broker.host}:{broker.port}')
                return broker

            # Per-instance method; the lookup reads current metadata every time.
            # It also covers coordinator lookups used by Kafka's producer.
            self.cluster.broker_metadata = mapped_lookup

    return CloudKafkaClient


def connection_config(bootstrap, address_map, client_id):
    return {
        'bootstrap_servers': bootstrap,
        'kafka_client': client_factory(bootstrap, address_map),
        'client_id': client_id,
        'allow_auto_create_topics': False,
        'security_protocol': 'PLAINTEXT',
        'api_version_auto_timeout_ms': 10000,
        'request_timeout_ms': 10000,
    }
