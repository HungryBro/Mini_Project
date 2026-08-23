# Mini Project — Taksin Bridge Vehicle Tracking

โค้ดหลักอยู่ใน `vehicle_tracking.py` และเครื่องมือเตรียม/เทรนข้อมูลอยู่ใน
`training/` ชุดข้อมูล โมเดล วิดีโอ และผลลัพธ์ทั้งหมดเป็นไฟล์ local จึงไม่ถูกส่งขึ้น
GitHub ตามกติกาใน `.gitignore`

โครงสร้างที่ใช้:

```text
vehicle_tracking.py                  ตัวตรวจจับและติดตาม
training/                            สคริปต์เตรียมข้อมูลและเทรน
taksin_bridge_sathorn/               วิดีโอ/ภาพต้นฉบับ (local)
data/                                dataset และ labels (local)
model/                               น้ำหนักโมเดล .pt (local)
runs/                                ผลเทรน/ผล inference (local)
graphify-out/                        รายงานวิเคราะห์ที่สร้างอัตโนมัติ (local)
archive/                             ไฟล์เก่าหรือเอกสารอ้างอิง (local)
```

Dataset ที่ใช้อยู่ตอนนี้คือ `data/taksin_vehicles/external/thai_cars_native/`
และคงคลาสเดิมจาก Thai-Cars ทั้งหมด ดูคำสั่งเทรนและใช้งานต่อได้ที่
[`training/README.md`](training/README.md)
