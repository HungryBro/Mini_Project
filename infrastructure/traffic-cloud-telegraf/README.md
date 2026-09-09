# Cloud MQTT → Telegraf → InfluxDB

Deployment package; creating these files does NOT deploy a cloud service.
Run this service on the Lab cloud server, not on the Mac. Kafka is not required.

Flow: Mac Tracker/Gateway → VerneMQ 172.16.2.117:1883 →
cloud Telegraf → InfluxDB mini_project → Grafana.

Input topic: v1/6610301004. Measurement: traffic_6610301004.
field_id is read from the gateway payload as a tag; expected value is 6610301004.
The separate measurement keeps this traffic stream distinct from other students.
All 22 field mappings and seven tag mappings retain the old Kafka parser's types.
complete_window stays boolean; use complete_window_value for numeric graphs.
Timestamp comes from payload.timestamp (Unix seconds), not ingestion time.

## Cloud administrator installation

Requires a Linux server with systemd, Telegraf supporting json_v2, the telegraf
user/group and executable at /usr/bin/telegraf. Verify these before installing.
Transfer this directory to the cloud through the administrator's approved channel.
Run the following commands in the transferred directory on the cloud:

```bash
sudo install -d -m 750 -o root -g telegraf /etc/traffic-telegraf
sudo install -m 640 -o root -g telegraf telegraf-mqtt.conf /etc/traffic-telegraf/telegraf-mqtt.conf
sudo install -m 600 -o root -g root .env.example /etc/traffic-telegraf/.env
sudoedit /etc/traffic-telegraf/.env
sudo install -m 644 traffic-telegraf.service /etc/systemd/system/traffic-telegraf.service
sudo systemctl daemon-reload
sudo systemctl enable --now traffic-telegraf
sudo journalctl -u traffic-telegraf -n 60 --no-pager
```

Set the token, organization, and bucket in .env. MQTT and Influx
connection values are read from the same file.
The token needs write permission on mini_project. Do not publish or commit it.
On updates, keep the existing .env instead of overwriting it with .env.example.

This dedicated service loads only its own configuration. Do not simultaneously
load this file into the Lab's existing Telegraf service: it would duplicate
subscribers and conflict with the fixed MQTT client ID.
If integrating into an existing agent instead, the administrator must merge
input/output routing explicitly to avoid sending other students' metrics to
this output or traffic metrics to other outputs.

## End-to-end verification

1. Start the cloud service before sending live traffic.
2. Run the existing Mac Gateway and V2 Tracker. Wait for a summary publish.
3. Check cloud logs for MQTT connection and absence of parse/write errors.
4. Paste verify.flux into InfluxDB Script Editor. Its _time must match the
   new summary; old rows alone are not proof of this deployment working.
5. After successful cloud verification, stop the old local traffic-telegraf
   writer to prevent two pipelines writing the same traffic. Verify a further
   fresh summary arrives with the old writer stopped.

An open MQTT/Influx port alone does not install or configure this subscriber.
There is no Telegraf HTTP upload port in this MQTT-input architecture.
Influx tokens authorize database access, not cloud service installation.

Persistent MQTT sessions help recover queued QoS 1 messages only after a
subscription has been established, subject to broker limits. Messages published
before subscription are not automatically saved for later replay.
Telegraf's default output buffer is in memory, not a disk-backed archive.

## Rollback

```bash
sudo systemctl disable --now traffic-telegraf
```

Then resume the previous writer if needed. Do not delete Influx data or Kafka
volumes as part of this change.
