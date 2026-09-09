"""Mac MQTT subscriber -> Cloud Kafka; Influx collector runs independently."""
from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka import errors as kafka_errors
from kafka.admin import KafkaAdminClient, NewTopic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mqtt.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_QOS, SUMMARY_TOPIC, STUDENT_ID,
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_BRIDGE_CLIENT_ID, KAFKA_ADDRESS_MAP,
)
from mqtt.kafka_connection import connection_config


def create_topic(bootstrap, topic):
    """Explicit setup step, creating only this project's topic if missing."""
    config = connection_config(bootstrap, KAFKA_ADDRESS_MAP, KAFKA_BRIDGE_CLIENT_ID + '_setup')
    config.pop('allow_auto_create_topics')  # This setting is for producer/consumer clients.
    admin = KafkaAdminClient(**config)
    try:
        response = admin.create_topics(
            [NewTopic(topic, num_partitions=1, replication_factor=1)], timeout_ms=10000,
        )
        for name, code, *detail in response.topic_errors:
            if code not in (0, kafka_errors.TopicAlreadyExistsError.errno):
                raise kafka_errors.for_code(code)(str(detail))
            print(f'[KAFKA TOPIC] {name}: ' + ('created' if code == 0 else 'already exists'), flush=True)
    finally:
        admin.close()


def check_brokers(bootstrap, topic):
    config = connection_config(bootstrap, KAFKA_ADDRESS_MAP, KAFKA_BRIDGE_CLIENT_ID + '_check')
    client = config.pop('kafka_client')(**config)
    try:
        client.add_topic(topic)
        future = client.cluster.request_update()
        client.poll(timeout_ms=10000, future=future)
        if not future.is_done:
            raise RuntimeError('Timed out reading Kafka metadata.')
        if future.failed():
            raise future.exception
        partitions = client.cluster.partitions_for_topic(topic)
        if not partitions:
            raise RuntimeError(f'Topic {topic} is missing or inaccessible.')
        for broker in client.cluster.brokers():
            target = client.cluster.broker_metadata(broker.nodeId)
            print(f'[KAFKA BROKER] id={broker.nodeId} advertised={broker.host}:{broker.port} '
                  f'connect={target.host}:{target.port}', flush=True)
            client.check_version(node_id=broker.nodeId, timeout=10)
        print(f'[KAFKA READY] topic={topic} partitions={len(partitions)}', flush=True)
    finally:
        client.close()


def verify_delivery(bootstrap, metadata, expected):
    """Read exactly the acknowledged record, without joining/committing a group."""
    config = connection_config(bootstrap, KAFKA_ADDRESS_MAP, KAFKA_BRIDGE_CLIENT_ID + '_verify')
    config['request_timeout_ms'] = 30000
    consumer = KafkaConsumer(group_id=None, enable_auto_commit=False, **config)
    partition = TopicPartition(metadata.topic, metadata.partition)
    try:
        consumer.assign([partition])
        consumer.seek(partition, metadata.offset)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            for records in consumer.poll(timeout_ms=500, max_records=1).values():
                for record in records:
                    if record.offset != metadata.offset:
                        raise RuntimeError('Kafka read-back returned a different offset.')
                    if record.value != expected or record.key != STUDENT_ID.encode():
                        raise RuntimeError('Kafka read-back did not match the original MQTT message.')
                    print(f'[KAFKA READ BACK OK] partition={record.partition} offset={record.offset} '
                          f'bytes={len(record.value)} exact_payload_match=true', flush=True)
                    return
        raise RuntimeError('Kafka acknowledged the write, but read-back timed out.')
    finally:
        consumer.close()


def validate_message(raw):
    body = json.loads(raw)
    if not isinstance(body, dict) or body.get('field_id') != STUDENT_ID:
        raise ValueError('Summary must belong to field_id=' + STUDENT_ID)
    if not isinstance(body.get('payload'), dict) or type(body.get('timestamp')) is not int:
        raise ValueError('Missing integer timestamp')
    if body['timestamp'] <= 0:
        raise ValueError('Invalid timestamp')
    if not isinstance(body.get('timestamp_th'), str) or not body['timestamp_th']:
        raise ValueError('Missing timestamp_th')
    return raw  # Preserve the exact JSON sent to MQTT; never attach the .env/token.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument('--topic', default=KAFKA_TOPIC)
    parser.add_argument('--check', action='store_true', help='Read broker metadata only.')
    parser.add_argument('--setup', action='store_true', help='Create this project topic if missing, check it, then exit.')
    parser.add_argument('--once', action='store_true', help='Exit after one Kafka delivery acknowledgement.')
    parser.add_argument('--verify', action='store_true', help='With --once, read back and compare the acknowledged record.')
    parser.add_argument('--timeout', type=float, default=90, help='Seconds to wait in --once mode.')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.verify and not args.once:
        parser.error('--verify requires --once')
    if args.setup:
        create_topic(args.bootstrap, args.topic)
    check_brokers(args.bootstrap, args.topic)
    if args.check or args.setup:
        return

    directory = Path(__file__).resolve().parents[2] / 'runs' / 'kafka_bridge'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'bridge.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another Kafka bridge is already running.')
    db = sqlite3.connect(directory / 'outbox.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY, topic TEXT, body BLOB)')
    producer = KafkaProducer(
        **connection_config(args.bootstrap, KAFKA_ADDRESS_MAP, KAFKA_BRIDGE_CLIENT_ID),
        enable_idempotence=True, acks='all', delivery_timeout_ms=20000,
        max_block_ms=10000,
    )
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=KAFKA_BRIDGE_CLIENT_ID, clean_session=False)
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    state = {'inflight': None, 'written': 0, 'retry_at': 0.0}

    def connected(client, userdata, flags, reason, properties):
        if reason != 0:
            print(f'[MQTT ERROR] {reason}', flush=True)
            return
        client.subscribe(SUMMARY_TOPIC, qos=MQTT_QOS)
        print(f'[MQTT IN] {MQTT_BROKER}:{MQTT_PORT} topic={SUMMARY_TOPIC}', flush=True)

    def received(client, userdata, message):
        try:
            body = validate_message(message.payload)
        except (ValueError, UnicodeError) as error:
            print(f'[IGNORED] {error}', flush=True)
            return
        # Commit before the MQTT callback returns and Paho acknowledges QoS 1.
        db.execute('INSERT INTO outbox(topic, body) VALUES (?, ?)', (args.topic, body))
        db.commit()
        print(f'[MQTT QUEUED] bytes={len(body)} field_id={STUDENT_ID}', flush=True)

    def send_next():
        if state['inflight'] or time.monotonic() < state['retry_at']:
            return
        row = db.execute('SELECT id, topic, body FROM outbox ORDER BY id LIMIT 1').fetchone()
        if row is None:
            return
        row_id, topic, body = row

        future = producer.send(topic, key=STUDENT_ID.encode(), value=body)
        state['inflight'] = (row_id, future, body)

    def finish_delivery():
        if state['inflight'] is None:
            return
        row_id, future, body = state['inflight']
        if not future.is_done:
            return
        state['inflight'] = None
        if future.failed():
            state['retry_at'] = time.monotonic() + 5
            print(f'[KAFKA ERROR] {future.exception}; retained in local outbox', flush=True)
            return
        metadata = future.value
        # Handle completion on the main thread, which owns the SQLite connection.
        db.execute('DELETE FROM outbox WHERE id=?', (row_id,))
        db.commit()
        state['written'] += 1
        print(f'[KAFKA OUT OK] topic={metadata.topic} partition={metadata.partition} '
              f'offset={metadata.offset} field_id={STUDENT_ID}', flush=True)
        if args.verify:
            verify_delivery(args.bootstrap, metadata, body)

    client.on_connect = connected
    client.on_message = received
    deadline = time.monotonic() + args.timeout
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        while not (args.once and state['written']):
            if args.once and time.monotonic() >= deadline:
                raise RuntimeError('Timed out without Kafka acknowledgement; queued data remains on disk.')
            code = client.loop(timeout=0.2)
            if code != mqtt.MQTT_ERR_SUCCESS:
                try:
                    client.reconnect()
                except OSError:
                    time.sleep(1)
            finish_delivery()
            if not (args.once and state['written']):
                send_next()
    except KeyboardInterrupt:
        print('Kafka bridge stopped.', flush=True)
    finally:
        client.disconnect()
        producer.close(timeout=5)
        print(f"Pending on disk: {db.execute('SELECT count(*) FROM outbox').fetchone()[0]}", flush=True)
        db.close()
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'[BRIDGE STOPPED] {error}', file=sys.stderr, flush=True)
        sys.exit(1)
