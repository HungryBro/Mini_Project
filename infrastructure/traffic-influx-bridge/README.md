# Traffic Influx Bridge

Telegraf bridge สำหรับงานตรวจรถสะพานกรุงธนโดยเฉพาะ:

```text
payload/mqtt/gateway.py
  -> Cloud VerneMQ 172.16.2.117:1883
  -> traffic-telegraf (Docker บน Mac)
  -> Cloud InfluxDB 172.16.2.117:8086 / bucket iot_data
  -> Grafana
```

ชุดนี้ไม่เปิด UDP และไม่สร้าง MQTT Broker, InfluxDB หรือ Grafana ซ้ำ
จึงใช้ร่วมกับ Cloud เดิมได้โดยตรง

## ตั้งค่าครั้งแรก

1. เปิด Docker Desktop แล้วรอให้สถานะเป็น Running
2. คัดลอกไฟล์ environment:

   ```bash
   cd "/Users/dolphin/Desktop/Mini Project/infrastructure/traffic-influx-bridge"
   cp .env.example .env
   ```

3. เปิด `.env` และแทนค่า `INFLUX_TOKEN` ด้วย API token ที่มีสิทธิ์เขียน
   bucket `iot_data` ของ InfluxDB `172.16.2.117:8086`
4. เริ่ม bridge:

   ```bash
   docker compose up -d
   docker compose logs -f traffic-telegraf
   ```

## ทดสอบ

เปิด MQTT gateway และ tracker ตามปกติ จากนั้นรอให้ gateway ส่ง Summary รอบใหม่
เมื่อ Telegraf เขียนสำเร็จ ให้เปิด InfluxDB Data Explorer แล้วเลือก:

```text
Bucket: iot_data
Measurement: traffic_summary
```

หรือใช้ Flux:

```flux
from(bucket: "iot_data")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "traffic_summary")
```

ค่าหลักที่ได้คือ `vehicle_count`, `wrong_way_count`,
`wrong_way_rate_per_100_vehicles`, `window_seconds` และ `complete_window`.
ทิศทางของทั้งสี่เลนถูกเก็บเป็น tags เพื่อเปรียบเทียบผลระหว่างรูปแบบ 2/2 และ 3/1

## หยุด bridge

```bash
docker compose down
```

อย่ารัน bridge นี้พร้อมกับ Telegraf Cloud ตัวอื่นที่เขียน traffic topic เดียวกัน
เว้นแต่ตั้งใจให้ทั้งสองตัวเขียนข้อมูลซ้ำ
