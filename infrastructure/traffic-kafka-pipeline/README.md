# Traffic MQTT -> Kafka -> InfluxDB Pipeline

ชุดนี้ดัดแปลง Kafka/Kafka Connect จากงาน IoT เดิมให้รับ payload ของ
Krung Thon Bridge โดยไม่เปลี่ยน `vehicle_tracking.py` หรือ MQTT Gateway

```text
vehicle_tracking.py
  -> payload/mqtt/gateway.py
  -> VerneMQ 172.16.2.117:1883
  -> Kafka Connect MQTT Source
  -> Kafka topic traffic.krung_thon_bridge.summary.v1
  -> Telegraf Kafka Consumer
  -> InfluxDB 172.16.2.117:8086 / mini_project
  -> Grafana
```

ชุดนี้ไม่สร้าง VerneMQ, InfluxDB หรือ Grafana ซ้ำ แต่รัน Kafka, Kafka Connect
และ Telegraf บน Mac ด้วย Docker Desktop

## 1. ตั้งค่าครั้งแรก

เปิด Docker Desktop แล้วเตรียม `.env`:

```bash
cd "/Users/dolphin/Desktop/Mini Project/infrastructure/traffic-kafka-pipeline"
cp .env.example .env
```

เปิด `.env` แล้วแทน `INFLUX_TOKEN` ด้วย token ที่อาจารย์ให้ และคง
`INFLUX_BUCKET=mini_project` ไว้ ห้ามส่ง token ลง Git หรือวางในแชต

## 2. เปิด Kafka pipeline

```bash
docker compose up -d
docker compose logs -f kafka-connect
```

ครั้งแรก Kafka Connect ต้องดาวน์โหลด MQTT Source Connector จึงอาจใช้เวลาสักครู่
เมื่อ REST API พร้อม ให้กด `Ctrl+C` เพื่อออกจากหน้า logs โดย container ยังทำงานอยู่

## 3. ลงทะเบียน MQTT -> Kafka Connector

คำสั่งนี้เรียกซ้ำได้ ถ้ามี connector อยู่แล้วจะอัปเดต config เดิม:

```bash
python3 -B register_connector.py
```

ผลที่ถูกต้องต้องเห็น connector และ task เป็น `RUNNING`

## 4. ดูข้อความใน Kafka

```bash
docker compose exec kafka kafka-console-consumer \
  --bootstrap-server kafka:29092 \
  --topic traffic.krung_thon_bridge.summary.v1 \
  --from-beginning
```

จากนั้นเปิด `payload/mqtt/gateway.py` และ `tracking/v2/vehicle_tracking.py`
ตามปกติ เมื่อ Summary รอบใหม่เข้า VerneMQ จะเห็น JSON เดียวกันใน Kafka

## 5. ตรวจ InfluxDB

Telegraf จะอ่าน Kafka และเขียน measurement `traffic_summary` ลง bucket
`mini_project` อัตโนมัติ ตรวจด้วย Flux:

```flux
from(bucket: "mini_project")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "traffic_summary")
```

ข้อมูลแต่ละชุดมี Influx tag `id = ID_6610301004` เพื่อแยกข้อมูลของงานนี้
ออกจากสมาชิกคนอื่นใน bucket เดียวกัน ส่วน `camera_id = CAM_112` ยังระบุกล้อง
ตามปกติ

ดู logs ของ Telegraf เมื่อข้อมูลไม่เข้า:

```bash
docker compose logs -f traffic-telegraf
```

## หยุดระบบ

```bash
docker compose down
```

ไม่ต้องเปิด `infrastructure/traffic-influx-bridge` พร้อมกัน เพราะชุดนั้นเป็นทางตรง
MQTT -> InfluxDB และจะข้าม Kafka
