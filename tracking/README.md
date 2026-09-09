# V2 Tracking — Krung Thon Bridge

ระบบที่ใช้งานจริงมีเฉพาะ V2: กล้อง 112 หนึ่งกล้อง, YOLO/ByteTrack สำหรับติดตามรถ,
และตารางเวลาเพื่อกำหนดทิศทางของ 4 เลน

```bash
cd "/Users/dolphin/Desktop/Mini Project/tracking/v2"
python3 -B vehicle_tracking.py
```

เลือก live stream หรือคลิปทดสอบได้ใน `tracking/v2/settings.py`.
V1 สามกล้องถูกเก็บอ้างอิงไว้ที่ `archive/tracking/v1/` และไม่ใช่ส่วนของ flow
ปัจจุบัน.
