# V2 — กล้อง 112 เท่านั้น

V2 ใช้กล้อง 112 เพียงตัวเดียวเสมอ: ตรวจจับและติดตามรถ, กำหนดทิศทางเลน
ด้วยตารางเวลา, ส่ง payload, เขียน JSONL และเมื่อเป็น live stream จะอัด MP4
พร้อมกันด้วย

ผลลัพธ์และ payload ของ V2 รวมชื่อวัตถุทุกชนิดเป็น `vehicle` เสมอ แต่ป้ายบน
ภาพจะแสดงสั้น ๆ เช่น `L2-V4` โดยเลข ID แยกตามเลน และจะพยายามคง ID เดิมเมื่อ
รถถูกบังระยะสั้น ๆ จน ByteTrack เปลี่ยนเลขชั่วคราว

การนับรถย้อนศรเป็นการนับ **เหตุการณ์** (`wrong_way_event_id`) ไม่ใช่นับเลข
ByteTrack ดิบ: รถต้องข้าม Wrong-Way Gate ใด Gate หนึ่งในทิศสวนเลน แล้วเคลื่อน
ต่อเนื่องหลังเส้นตามค่าที่ตั้งใน `settings.py` จึงถูกนับครั้งเดียวต่อ ID. Gate 1–3
ช่วยให้รถที่เริ่มถูก track หลังเส้นหนึ่ง ยังมีโอกาสถูกยืนยันจากเส้นถัดไป โดยไม่
กลับไปพึ่งการสั่นของกล่องตรวจจับเพียงอย่างเดียว

เลือกแหล่งวิดีโอใน `settings.py` ของโฟลเดอร์นี้:

```python
SOURCE_MODE = "live_stream"  # กล้องสด
# หรือ
SOURCE_MODE = "video_files"  # ไฟล์ CAMERA_112_FILE
```

วิธีรัน:

```bash
cd "/Users/dolphin/Desktop/Mini Project/tracking/v2"
python3 vehicle_tracking.py
```

กด `q` เพื่อหยุด. V2 ไม่มีการเปิดหรืออ่านกล้องอื่น.

## MQTT แบบแยก Gateway

V2 ไม่ใช้ UDP แล้ว โดย Tracker จะส่ง Gateway-input ทุก 1 นาทีไปยัง MQTT topic
`traffic/krung_thon_bridge/CAM_112/gateway_input` ส่วน Gateway ที่รันแยกจะรับ
ข้อมูลนี้ แสดง JSON เต็ม และรวมเป็น Cloud summary ทุก 5 นาทีไปยัง topic
`traffic/krung_thon_bridge/CAM_112/summary`

เปิด Gateway ก่อนใน Terminal แรก:

```bash
cd "/Users/dolphin/Desktop/Mini Project/payload"
python3 mqtt/gateway.py
```

จากนั้นเปิด Tracker ใน Terminal ที่สองตามคำสั่งด้านบน. หากต้องการดูเฉพาะ
summary ที่ Gateway ส่งออก ให้เปิด Terminal ที่สามและรัน:

```bash
cd "/Users/dolphin/Desktop/Mini Project/payload"
python3 mqtt/broker_receiver.py
```

ค่า Broker และ MQTT topics อยู่ที่ `payload/mqtt/settings.py`.
