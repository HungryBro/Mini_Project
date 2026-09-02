# Payload — Krung Thon Bridge

โฟลเดอร์นี้แบ่งตามหน้าที่ ไม่ปนกัน:

```text
payload/
├── mqtt/
│   ├── settings.py          # Broker, topics และ MQTT IDs (ที่ใช้งานจริง)
│   ├── gateway.py           # Gateway: รับ 15 วินาที → รวม/ส่ง Summary 1 นาที
│   └── broker_receiver.py   # ดู Summary ที่ Broker (ไม่บังคับ)
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
            └─ every 1 minute → traffic/krung_thon_bridge/CAM_112/summary
```

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
