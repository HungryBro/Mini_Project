# Payload — Krung Thon Bridge

Kafka Cloud: ดู [วิธีรันและสถานะการเชื่อมต่อ](mqtt/KAFKA.md).

Worker A วิเคราะห์ความสัมพันธ์ระหว่างเลน 2/2 กับ 3/1 และรถย้อนศร:
ดู [วิธีรัน ตรวจสอบ และแสดงผลใน Grafana](ml/README.md).

โฟลเดอร์นี้แบ่งตามหน้าที่ ไม่ปนกัน:

```text
payload/
├── mqtt/
│   ├── settings.py          # Broker, topics และ MQTT IDs (ที่ใช้งานจริง)
│   ├── gateway.py           # Gateway: รับ 15 วินาที → รวม/ส่ง Summary 1 นาที
│   ├── broker_receiver.py   # ดู Summary ที่ Broker (ไม่บังคับ)
│   ├── influxdb_collector.py # MQTT → InfluxDB Cloud
│   ├── kafka_bridge.py      # MQTT → Kafka Cloud
│   └── kafka_connection.py  # แปลงที่อยู่ broker เฉพาะ Kafka client
├── common/
│   └── traffic_payload.py   # รูปแบบ payload + ตัวรวมข้อมูลที่ MQTT Gateway ใช้
└── tools/
    └── replay_traffic_payload.py  # เล่น JSONL เก่าย้อนหลัง (เครื่องมือเสริม)

archive/
└── udp/                     # UDP เก่า ไม่ถูกใช้ใน V2 MQTT flow
```

V2 ใช้ MQTT เท่านั้น:

```text
Tracker (camera 112)
  └─ every 15 seconds → traffic/krung_thon_bridge/CAM_112/gateway_input
       └─ MQTT Gateway
            └─ every 1 minute → v1/6610301004
```

`v1/6610301004` เป็น summary topic ที่ collector และ Kafka bridge ของเรา subscribe;
เลข v1 ไม่เกี่ยวกับ V1 tracker ที่ถูกเก็บไว้ใน archive.

## Terminal 1 — MQTT Gateway

เปิดก่อนเสมอ เพื่อรับ Gateway-input, แสดง JSON เต็มทุก 15 วินาที และส่ง
summary ออกทุก 1 นาที:

```bash
cd "/Users/dolphin/Desktop/Mini Project/payload"
python3 mqtt/gateway.py
```

## Terminal 2 — Tracker

```bash
cd "/Users/dolphin/Desktop/Mini Project/tracking/v2"
python3 vehicle_tracking.py
```

## Terminal 3 — ดูเฉพาะ Summary ที่ Broker (ไม่บังคับ)

```bash
cd "/Users/dolphin/Desktop/Mini Project/payload"
python3 mqtt/broker_receiver.py
```

แก้ Broker, port และ topics ที่ `mqtt/settings.py` เพียงไฟล์เดียว. หาก Gateway
หรือ Tracker แจ้ง timeout ให้เชื่อมต่อเครือข่ายที่เข้าถึง Broker
`172.16.2.117:1883` ก่อน.

## สถานะไฟล์

`common/traffic_payload.py` ยังใช้งานอยู่โดยทั้ง Tracker และ MQTT Gateway
จึงไม่ย้ายไป archive. ส่วน `archive/udp/` เป็นโค้ด UDP เก่าที่เก็บไว้เผื่อ
อ้างอิงเท่านั้น และไม่ถูกเรียกใช้ใน V2.

## Terminal 3 — Direct InfluxDB Collector

เปิดตัวนี้พร้อม Gateway และ Tracker เพื่อรับ summary จาก VerneMQ แล้วเขียน
เข้า InfluxDB Cloud โดยตรง:

~~~bash
cd "/Users/dolphin/Desktop/Mini Project"
python3 -B payload/mqtt/influxdb_collector.py
~~~

ข้อมูลเข้า bucket mini_project, measurement traffic_6610301004 และ tag
field_id=6610301004. ค่าเชื่อมต่ออยู่ใน infrastructure/traffic-cloud-telegraf/.env
และไม่ถูกเก็บใน Git. เติม --once ท้ายคำสั่งเพื่อทดสอบเพียงหนึ่ง summary.
