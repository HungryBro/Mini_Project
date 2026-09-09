# Mini Project — Krung Thon Bridge V2

ระบบปัจจุบันใช้กล้อง 112 กล้องเดียวเพื่อตรวจจับและติดตามรถบนสะพานกรุงธน
ทิศทางของทั้ง 4 เลนมาจากตารางเวลาโดยตรง ไม่ได้อ่านไฟจราจรจากกล้องอื่น

```text
Camera 112 / YOLO + ByteTrack
  → VerneMQ (gateway_input ทุก 15 วินาที)
  → MQTT Gateway (รวม Summary 1 นาที)
  → VerneMQ (v1/6610301004)
      ├─ Influx Collector → InfluxDB mini_project → Grafana
      └─ Kafka Bridge → Kafka Cloud 172.16.2.117:9092
```

## รันระบบ

เปิด MQTT Gateway ก่อน:

```bash
cd "/Users/dolphin/Desktop/Mini Project/payload"
python3 -B mqtt/gateway.py
```

แล้วเปิด Tracker V2:

```bash
cd "/Users/dolphin/Desktop/Mini Project/tracking/v2"
python3 -B vehicle_tracking.py
```

รายละเอียดแหล่งวิดีโอ โมเดล และตารางเวลาอยู่ใน `tracking/v2/settings.py`.

เปิดอีก Terminal สำหรับส่งเข้า InfluxDB:

```bash
cd "/Users/dolphin/Desktop/Mini Project"
python3 -B payload/mqtt/influxdb_collector.py
```

เปิดอีก Terminal สำหรับส่งเข้า Kafka Cloud:

```bash
cd "/Users/dolphin/Desktop/Mini Project"
python3 -B payload/mqtt/kafka_bridge.py
```

ใช้งานทั้งสองปลายทางพร้อมกันได้โดยไม่ต้องเปิด Docker.
รายละเอียดการเชื่อมต่อและทดสอบอ่านข้อมูลกลับอยู่ใน [คู่มือ Kafka](payload/mqtt/KAFKA.md).

## ข้อมูลสำหรับ InfluxDB และ Grafana

ข้อมูลใหม่เขียนลง bucket `mini_project`, measurement `traffic_6610301004` และ tag
`field_id=6610301004`.

ค่าที่ทำกราฟได้ เช่น `vehicle_count`, `wrong_way_count`, ข้อมูลรถแต่ละเลน และ
`lane_N_direction_value` โดยกำหนด `up=1`, `down=0`. ค่าทิศข้อความยังเป็น tag
สำหรับกรองข้อมูล.

## Archive

โค้ด V1 แบบสามกล้องและการอ่านสัญญาณไฟถูกย้ายไปที่ `archive/tracking/v1/` เพื่อ
อ้างอิงเท่านั้น และไม่ใช่ส่วนของระบบ V2 ที่ใช้งานจริง.
