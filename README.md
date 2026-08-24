# Mini Project — Bridge Vehicle Tracking

โค้ดหลักอยู่ใน `vehicle_tracking.py` และเครื่องมือเตรียม/เทรนข้อมูลอยู่ใน
`training/` ชุดข้อมูล โมเดล วิดีโอ และผลลัพธ์ทั้งหมดเป็นไฟล์ local จึงไม่ถูกส่งขึ้น
GitHub ตามกติกาใน `.gitignore`

โครงสร้างที่ใช้:

```text
vehicle_tracking.py                  ตัวตรวจจับและติดตาม
training/                            สคริปต์เตรียมข้อมูลและเทรน
locations/
  taksin/
    image/                           ภาพต้นฉบับตากสิน (local)
    video/                           วิดีโอต้นฉบับตากสิน (local)
  prachanukul/
    image/                           ภาพต้นฉบับประชานุกูล (local)
    video/                           วิดีโอต้นฉบับประชานุกูล (local)
data/                                dataset และ labels (local)
model/
  taksin/                            โมเดลที่เทรนจาก Thai-Cars (local)
  coco/                              YOLO มาตรฐาน COCO (local)
runs/
  taksin/image/, taksin/video/       ผลรันตากสิน (local)
  prachanukul/image/, prachanukul/video/  ผลรันประชานุกูล (local)
graphify-out/                        รายงานวิเคราะห์ที่สร้างอัตโนมัติ (local)
archive/                             ไฟล์เก่าหรือเอกสารอ้างอิง (local)
```

Dataset ที่ใช้อยู่ตอนนี้คือ `data/taksin_vehicles/external/thai_cars_native/`
โดยคงคลาสรถเดิมจาก Thai-Cars และตัด `human` ออก ดูคำสั่งเทรนและใช้งานต่อได้ที่
[`training/README.md`](training/README.md)

## การรันตามสถานที่

ตากสินใช้โมเดลที่เทรนเองและกรอบสะพานเดิม:

```bash
python vehicle_tracking.py \
  --model model/taksin/yolo11n_native/weights/best.pt \
  --source locations/taksin/video/taksin_bridge_sathorn_1min.mp4 \
  --output-dir runs/taksin/video/custom_model \
  --merge-car-like --no-show-gates
```

`--merge-car-like` รวม `taxi`, `van` และ `pickup` เป็น `car` ในผลลัพธ์ โดยไม่ต้องเทรนใหม่

ประชานุกูลใช้ YOLO มาตรฐาน COCO เฉพาะ 4 คลาส และปิดกรอบสะพานของตากสิน เพราะ
มุมกล้องต่างกัน:

```bash
python vehicle_tracking.py \
  --model model/coco/yolo11m.pt \
  --source locations/prachanukul/video/prachanukul_ratchavipha_1min_timestamp_font16.mp4 \
  --output-dir runs/prachanukul/video/coco_yolo11m \
  --classes car,bus,truck,motorcycle \
  --no-bridge-only --no-show-gates
```
