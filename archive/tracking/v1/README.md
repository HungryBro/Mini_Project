# V1 — ระบบเดิมสามกล้อง

V1 ใช้กล้อง 112 ตรวจจับ/ติดตามรถ และใช้กล้อง 147 กับ 156 อ่านไฟจราจร
เพื่อกำหนดทิศทางของทั้ง 4 เลน โดยมีตารางเวลาเป็น fallback เมื่อไฟไม่ชัด

ตั้งค่าของ V1 อยู่ใน `settings.py` ของโฟลเดอร์นี้:

```python
SOURCE_MODE = "live_stream"  # หรือ "video_files"
```

วิธีรัน:

```bash
cd "/Users/dolphin/Desktop/Mini Project/tracking/v1"
python3 vehicle_tracking.py
```

กด `q` เพื่อหยุด V1. V1 และ V2 แยก settings/ไฟล์รันออกจากกันแล้ว

