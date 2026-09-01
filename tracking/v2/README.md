# V2 — กล้อง 112 เท่านั้น

V2 ใช้กล้อง 112 เพียงตัวเดียวเสมอ: ตรวจจับและติดตามรถ, กำหนดทิศทางเลน
ด้วยตารางเวลา, ส่ง payload, เขียน JSONL และเมื่อเป็น live stream จะอัด MP4
พร้อมกันด้วย

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
