# Kafka Cloud — 6610301004

ส่งข้อมูลเข้า Kafka ที่ `172.16.2.117:9092` ได้จากเครื่อง Mac โดยไม่ต้องเปิด Docker
หรือติดตั้งบริการเพิ่มบน Cloud:

```text
Tracker → VerneMQ (gateway_input) → Gateway → VerneMQ (v1/6610301004)
                                              ├─ influxdb_collector.py → InfluxDB → Grafana
                                              └─ kafka_bridge.py → Kafka Cloud
```

Gateway, Tracker, Influx collector และ Kafka bridge รันบน Mac ส่วน VerneMQ,
Kafka และ InfluxDB อยู่บน Cloud. Kafka เป็นอีกทางออกของ summary เดียวกัน.

## รันใช้งาน

ติดตั้ง Kafka client ครั้งแรก (เครื่องนี้ติดตั้งแล้ว):

```bash
python3 -m pip install 'paho-mqtt>=2.1,<3' 'kafka-python==2.2.15'
```

เปิด Gateway และ Tracker ตาม README หลัก แล้วเปิด Terminal เพิ่มสำหรับ Kafka:

```bash
cd "/Users/dolphin/Desktop/Mini Project"
python3 -B payload/mqtt/kafka_bridge.py
```

เมื่อ Gateway ส่ง summary จะเห็น:

```text
[MQTT QUEUED] bytes=... field_id=6610301004
[KAFKA OUT OK] topic=traffic.krung_thon_bridge.summary.v1 partition=0 offset=... field_id=6610301004
```

`KAFKA OUT OK` หมายถึง Kafka ยืนยันรับแล้ว ไม่ใช่แค่เชื่อมต่อสำเร็จ.
กด Ctrl+C เพื่อหยุด. หากต้องการให้ข้อมูลเข้า InfluxDB ด้วย ให้เปิด
`python3 -B payload/mqtt/influxdb_collector.py` อีก Terminal ตามเดิม.

## ค่าเชื่อมต่อ

ตั้งค่าที่ `payload/mqtt/settings.py`:

- MQTT: `172.16.2.117:1883`, topic `v1/6610301004`
- Kafka: `172.16.2.117:9092`, topic `traffic.krung_thon_bridge.summary.v1`
- Kafka record key: `6610301004`
- Kafka record value: JSON จาก MQTT ทั้งก้อน โดยเก็บ bytes และชนิดข้อมูลเดิม

Kafka ไม่ใช้ Influx token. ตัว bridge ไม่โหลด `.env` ของ InfluxDB.
เลข `v1` ใน topic เป็นรุ่นรูปแบบข้อมูล ไม่ใช่ Tracker V1.

Cloud แจ้ง broker 1 เป็น `localhost:9092` ใน metadata. ตัวเชื่อมของเราจึงใช้
`KAFKA_ADDRESS_MAP` แปลงที่อยู่นี้เป็น `172.16.2.117:9092` เฉพาะ Kafka client
ในโปรแกรม ผ่าน `kafka_client` extension point ของ kafka-python.
การ lookup ทุกครั้งใช้ metadata ล่าสุด จึงยังทำงานหลัง metadata refresh.
ไม่ได้แก้ hosts/DNS ของ Mac และไม่ได้เปิด local broker หรือ proxy.

ถ้า Cloud เปลี่ยนมาประกาศ IP โดยตรง mapping นี้จะไม่ถูกใช้.
หากเปลี่ยน bootstrap IP ต้องแก้ mapping ให้ตรงกันด้วย; ไม่ใช้ mapping เดียว
แทน broker หลายตัว. Pin kafka-python 2.2.15 เพราะ adapter อาศัยวิธี lookup
broker ของรุ่นนี้. ใช้ PLAINTEXT ตาม endpoint ปัจจุบันของ Lab.

## ตรวจและทดสอบ

ตรวจ metadata พร้อมการเชื่อมต่อ broker โดยไม่ส่งข้อมูล:

```bash
python3 -B payload/mqtt/kafka_bridge.py --check
```

ส่ง summary จริงหนึ่งรายการแล้วอ่านกลับจาก partition/offset เดียวกัน:
หยุด Kafka bridge ตัวเดิมก่อน แล้วรันขณะที่ Gateway และ Tracker ทำงาน:

```bash
python3 -B payload/mqtt/kafka_bridge.py --once --verify --timeout 90
```

เมื่อสำเร็จจะเห็นทั้ง `KAFKA OUT OK` และ `KAFKA READ BACK OK`.
การอ่านกลับไม่เข้าร่วม consumer group และไม่ commit offset ของผู้ใช้อื่น.
`--once` อาจส่งรายการค้างใน outbox ก่อน summary ใหม่; หากไม่มีรายการค้าง
ต้องรอรอบ summary จาก Gateway. `--verify` ต้องมีสิทธิ์อ่าน topic ด้วย.

ทดสอบจริงวันที่ 9 กันยายน 2026: รับ MQTT summary 873 bytes ของ `6610301004`,
Kafka ยืนยันที่ partition `0`, offset `0` แล้วอ่านกลับเทียบทั้ง key และ
payload ได้ตรงทุก byte. หลังทดสอบไม่มีรายการค้างบนดิสก์.

Topic นี้สร้างแล้ว (1 partition, replication factor 1 ตาม single-broker Lab).
หากย้ายไปคลัสเตอร์ใหม่และยังไม่มี topic ใช้คำสั่งตั้งค่าครั้งเดียว:

```bash
python3 -B payload/mqtt/kafka_bridge.py --setup
```

คำสั่งนี้สร้างเฉพาะ topic ที่ตั้งค่าไว้หากยังไม่มี ไม่แก้ topic เดิม.
การรันปกติและ `--check` ไม่สร้าง topic อัตโนมัติ.

## ข้อมูลค้างและการส่งซ้ำ

ข้อมูล MQTT ที่รับแล้วบันทึกลง `runs/kafka_bridge/outbox.sqlite3` ก่อนส่ง Kafka.
ลบแต่ละรายการหลัง Kafka ยืนยันรับ. หากส่งล้มเหลว รายการยังอยู่เพื่อ retry
หรือส่งต่อเมื่อเปิดโปรแกรมใหม่. มี lock ป้องกันเปิด bridge ซ้ำในเครื่องนี้.

การส่งเป็น at-least-once: อาจซ้ำหากโปรแกรมหยุดระหว่าง Kafka รับสำเร็จและลบ
รายการบนดิสก์ หรือ MQTT ส่งซ้ำ. Producer idempotence ไม่ขจัดการซ้ำข้าม restart.
Consumer ควรใช้ `field_id`, `place_id`, `timestamp` เป็นคีย์ตรวจซ้ำ.
Persistent MQTT session ไม่ได้กู้ข้อมูลก่อนการ subscribe ครั้งแรก.
ให้เปิด bridge ค้างไว้ระหว่างใช้งาน; การทดสอบ `--once` จบแล้วจะหยุดโปรแกรม.

Local regression checks:

```bash
python3 -B -m unittest discover -s payload/tests -p 'test_kafka_connection.py' -v
```

Client extension reference: [KafkaProducer documentation](https://kafka-python.readthedocs.io/en/2.2.15/apidoc/KafkaProducer.html).
