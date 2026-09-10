# Worker A: การจัดเลน 2/2 กับ 3/1 และรถย้อนศร

เส้นทาง: Kafka Cloud → `lane_effect_worker.py` บน Mac → InfluxDB Cloud → Grafana

ใช้ topic ปัจจุบันจาก `payload/mqtt/settings.py` และ group แยก
`traffic_lane_effect_6610301004_v1` ไม่แย่งข้อความกับ worker อื่น
ครั้งแรกอ่านตั้งแต่ข้อความเก่าสุดที่ Kafka ยังเก็บไว้ ครั้งถัดไปต่อจาก offset เดิม
Kafka retention จำกัดข้อมูลย้อนหลัง: เปิด worker เก็บต่อเนื่องเพื่อสะสมประวัติใน SQLite

## เปิดใช้งาน

เปิด Tracker, Gateway และ Kafka bridge ตามเดิม ส่วน direct Influx collector
ยังใช้เก็บข้อมูลดิบ เปิด Terminal เพิ่มอีกหน้าสำหรับ Worker A:

```bash
cd "/Users/dolphin/Desktop/Mini Project"
python3 -m pip install 'statsmodels>=0.14.4,<0.15'
python3 -B payload/ml/lane_effect_worker.py
```

อ่าน Influx URL, org, bucket และ token จาก
`infrastructure/traffic-cloud-telegraf/.env` เดิม ตรวจให้ `INFLUX_BUCKET=mini_project`
ไม่มี Docker เพิ่ม และไม่ใช้ไฟล์ร่าง Isolation Forest `traffic_ml_worker.py`

Log ที่คาดหวัง:

```text
[ML A START] kafka=172.16.2.117:9092 topic=traffic_6610301004 ...
[ML A RECORD] timestamp=... mode=0 status=collecting
[ML A INFLUX OK] points=2 bucket=mini_project
```

`mode=0` คือ 2/2, `mode=1` คือ 3/1 เฉพาะรูปแบบ [up,up,down,down]
และ [up,up,up,down] เท่านั้น; รูปแบบอื่นจะถูกบันทึกเป็น rejected
Ctrl+C หยุดได้ เปิดคำสั่งเดิมเพื่อทำต่อ

## ข้อมูลใน InfluxDB

ทั้งสอง measurement มี tag `field_id=6610301004`, `place_id`, `model_version`:

| Measurement | ความหมาย |
|---|---|
| `traffic_lane_observed_6610301004` | จำนวนรถ/ย้อนศรจริง อัตรารวม อัตราเลน 3 และโหมดเลน ทุก summary |
| `traffic_lane_model_6610301004` | สถิติสะสมและผล Binomial regression คำนวณใหม่ทุก 10 นาทีของข้อมูล |

ข้อมูลดิบ `traffic_6610301004` ยังมาจาก collector เดิม
Worker A คำนวณร้อยละเอง ไม่เพิ่ม rate กลับเข้า MQTT payload
จำนวนรถเป็นศูนย์จะไม่มี field อัตรา และ `lane_3_rate_valid=0` (ไม่แทนด้วยอัตรา 0%)
อัตราสะสมใช้ผลรวมย้อนศร / ผลรวมรถ ไม่ใช้ค่าเฉลี่ยของเปอร์เซ็นต์แต่ละนาที

## วิธีตรวจสอบหลังติดตั้ง

1. ทดสอบในเครื่อง ไม่เชื่อม Cloud:

```bash
python3 -B -m unittest discover -s payload/tests -p 'test_lane_analysis.py' -v
```

2. ลองรับข้อความจริงหนึ่งชุดและเขียน Influx (จะใช้ข้อความเก่าถ้ามี backlog):

```bash
python3 -B payload/ml/lane_effect_worker.py --once --timeout 120
```

ให้หยุด Worker A ตัวเดิมก่อนเรียก `--once` เพราะมี lock ป้องกันเปิดซ้ำ
เมื่อผ่านแล้วเปิดคำสั่งปกติให้ทำงานต่อเนื่อง

3. ตรวจจำนวนข้อมูลสะสมและคิวในเครื่องได้แม้ worker กำลังเปิดอยู่:

```bash
python3 -B payload/ml/lane_effect_worker.py --status
```

`observations` ควรเพิ่ม, `outbox=0` หลังเขียนสำเร็จ, `rejected` ควรเป็น 0
ถ้า rejected เพิ่มให้ดู `[ML A REJECTED]` ใน Terminal รายละเอียดเก็บใน SQLite
ที่ `runs/ml_a/history.sqlite3` (ไม่ขึ้น Git)

4. ใน Influx Data Explorer เปิด Script Editor วาง query นี้:

```flux
from(bucket: "mini_project")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "traffic_lane_observed_6610301004")
  |> filter(fn: (r) => r.field_id == "6610301004")
  |> filter(fn: (r) => r._field == "lane_3_wrong_way_rate_pct")
  |> last()
```

ตรวจ `_time` ให้ตรงกับเวลา summary ใน Terminal ไม่ใช่เวลาที่ส่งใหม่
ข้อมูล Kafka เก่าอาจอยู่นอก Past 15m ให้ขยายช่วงตามเวลาข้อความ
หากเลน 3 ไม่มีรถให้ลอง `_field == "lane_3_vehicle_count"` แทน

ตรวจสถานะโมเดลด้วย query เดียวกัน เปลี่ยน measurement เป็น
`traffic_lane_model_6610301004` และ `_field` เป็น `model_ready`

5. Grafana (Influx data source ภาษา Flux) ใช้:

```flux
from(bucket: "mini_project")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "traffic_lane_observed_6610301004")
  |> filter(fn: (r) => r.field_id == "6610301004")
  |> filter(fn: (r) => r._field == "lane_3_wrong_way_rate_pct")
```

แสดงเปอร์เซ็นต์และสร้างอีก panel สำหรับ `lane_mode_31` ใช้ value mapping 0→2/2, 1→3/1
ผลโมเดลแสดง `observed_rate_22_pct`, `observed_rate_31_pct`,
`adjusted_odds_ratio`, `odds_ratio_ci_low`, `odds_ratio_ci_high` และ `model_ready`
ตั้ง Stat ไม่ให้ใช้ผลเก่านอกช่วงเวลาที่เลือก; ดู tag `status` คู่กับตัวเลขเสมอ

## เกณฑ์และการตีความ

- วิเคราะห์เลน 3 เป็นหลัก ด้วย Binomial GLM: [ย้อนศร, รถทั้งหมด−ย้อนศร]
- ตัวแปรอธิบาย: โหมดเลน, เวลาแบบ sin/cos ของ 24 ชั่วโมง และวันในสัปดาห์
- ใช้ข้อมูลย้อนหลัง 30 วันแยกตามสถานที่ และ standard errors แบบ cluster ตามวัน
- ต้องมีอย่างน้อย 14 วัน, 60 ช่วงที่มีรถต่อโหมด, อย่างน้อย 10 รถย้อนศรและ
  10 รถไม่ย้อนศรต่อโหมด พร้อมตัวแปรที่แยกจากกันได้ ก่อน fit
  นี่เป็นเกณฑ์เริ่มต้นเชิงปฏิบัติ ไม่ใช่การรับประกันความแม่นยำหรือ power ของการศึกษา
- `collecting`: ยังสะสมข้อมูล; `insufficient_events`: เหตุการณ์ไม่พอ;
  `confounded_design`: ตัวแปรแยกกันไม่ได้; `fit_failed` / `unstable_fit`: ยังไม่รายงานค่าประมาณ
- `exploratory`: fit ได้และมีทั้งสองโหมดในบางช่องวันในสัปดาห์/ชั่วโมงเดียวกัน
- `time_confounded`: fit ได้แต่ไม่มีช่องเวลาเหลื่อมกัน ผลพึ่งสมมติฐานรูปแบบเวลาอย่างมาก
- `model_ready=1` แปลว่า fit สำเร็จ ไม่ใช่ยืนยันสมมติฐานว่ารถย้อนศรเพิ่ม
- odds ratio >1 หมายถึง odds ย้อนศรสัมพันธ์กับโหมด 3/1 มากกว่า 2/2
  ไม่ใช่จำนวนรถเพิ่มเท่านั้นและไม่ใช่ risk ratio; ช่วงความเชื่อมั่นคร่อม 1 คือ
  ข้อมูลยังไม่แสดงความแตกต่างชัดภายใต้โมเดล ไม่ใช่พิสูจน์ว่าไม่มีผล
- `fitted_rate_*` เป็นค่าฟิตเปรียบเทียบสองโหมดบนข้อมูลฝึกชุดเดียวกัน
  ไม่ใช่พยากรณ์อนาคตหรือคะแนนความแม่นยำจากข้อมูลทดสอบ
- โมเดลนี้ยังไม่รวมสภาพอากาศ/เหตุการณ์อื่น และไม่พิสูจน์เหตุและผล
  ต้องตรวจเวลาจัดเลนจริงกับตารางและตรวจคลิปย้อนหลังใกล้เวลาเปลี่ยนเลน
  ก่อนนำผลไปสรุปงานวิจัย ตอนนี้ payload ไม่มีเวลาต้นช่วง/เวลาหลังเปลี่ยนเลน
  จึงยังไม่วิเคราะห์หน้าต่างก่อน–หลังเปลี่ยนเลนแบบละเอียด

## การส่งซ้ำและการกู้คืน

SQLite บันทึกประวัติพร้อม outbox ใน transaction ก่อน commit Kafka offset
ถ้า Influx ล้มเหลว worker หยุดพร้อมเก็บคิว ให้เปิดคำสั่งเดิมเพื่อส่งต่อ
การเขียนซ้ำใช้ measurement/tags/เวลาเดิม จึงไม่สร้างจุดซ้ำใหม่
ใช้ place_id+timestamp ตรวจ summary ซ้ำ; ถ้าค่าไม่ตรงกันจะ quarantine แทนการเลือกทับ
รองรับข้อมูลหนึ่งกล้องต่อสถานที่ตาม payload ปัจจุบัน ถ้าเพิ่มกล้องต้องเพิ่ม camera ID ใน key
อย่าลบ SQLite หลัง Kafka commit แล้ว โดยเฉพาะเมื่อ outbox ยังมีข้อมูล
ประวัติใน SQLite เก็บต่อเนื่อง; model เลือกเพียง 30 วันล่าสุดตามเวลา event
ข้อมูลมาช้าจะเก็บค่าจริง แต่รอรอบวิเคราะห์ถัดไป ไม่เขียนทับผลวิเคราะห์ในอดีต

เอกสารอ้างอิง:
- https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html
- https://kafka-python.readthedocs.io/en/2.2.17/apidoc/KafkaConsumer.html
