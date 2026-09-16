# V2 Tracking — Krung Thon Bridge

ระบบที่ใช้งานจริงมีเฉพาะ V2: กล้อง 112 หนึ่งกล้อง, YOLO/ByteTrack สำหรับติดตามรถ,
และตารางเวลาเพื่อกำหนดทิศทางของ 4 เลน

```bash
cd "/Users/dolphin/Desktop/Mini Project/tracking/v2"
python3 -B vehicle_tracking.py
```

### การ Replay ข้อมูลย้อนหลัง (แบบที่ 1 — ไม่ต้องใช้กล้องสด):
กำหนด `SOURCE_MODE = "replay"` ใน [`tracking/v2/settings.py`](file:///Users/dolphin/Desktop/Mini%20Project/tracking/v2/settings.py) แล้วสั่งรัน `python3 vehicle_tracking.py` หรือรันผ่าน helper script:
```bash
python3 tracking/replay_option1.py
```

เลือก live stream, คลิปทดสอบ หรือโหมด replay ได้ใน `tracking/v2/settings.py`.
V1 สามกล้องถูกเก็บอ้างอิงไว้ที่ `archive/tracking/v1/` และไม่ใช่ส่วนของ flow ปัจจุบัน.
