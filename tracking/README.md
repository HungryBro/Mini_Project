# Tracking entry points

เลือกใช้เพียงโฟลเดอร์เดียวต่อครั้ง:

```bash
# V1: กล้อง 112 + 147 + 156 และอ่านสัญญาณไฟ
cd "/Users/dolphin/Desktop/Mini Project/tracking/v1"
python3 vehicle_tracking.py

# V2: กล้อง 112 เท่านั้น; เลือกไฟล์หรือ live stream ใน v2/settings.py
cd "/Users/dolphin/Desktop/Mini Project/tracking/v2"
python3 vehicle_tracking.py
```

อย่ารันไฟล์ที่ระดับ `tracking/` เพราะรันแยกตาม v1 หรือ v2 แล้ว.
