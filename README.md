# Mini Project — Bridge Vehicle Tracking

โค้ดหลักอยู่ใน `vehicle_tracking.py` และเครื่องมือเตรียม/เทรนข้อมูลอยู่ใน
`training/` ชุดข้อมูล โมเดล วิดีโอ และผลลัพธ์ทั้งหมดเป็นไฟล์ local จึงไม่ถูกส่งขึ้น
GitHub ตามกติกาใน `.gitignore`

โครงสร้างที่ใช้:

```text
vehicle_tracking.py                  ตัวตรวจจับและติดตาม
training/                            สคริปต์เตรียมข้อมูลและเทรน
locations/
  krung_thon_bridge/
    video/                           วิดีโอต้นฉบับกรุงธน (local)
data/                                dataset และ labels (local)
model/
  taksin/                            โมเดลที่เทรนจาก Thai-Cars (local)
  coco/                              YOLO มาตรฐาน COCO (local)
runs/
  krung_thon_bridge/                 ผลรันกรุงธน (local)
graphify-out/                        รายงานวิเคราะห์ที่สร้างอัตโนมัติ (local)
archive/                             ไฟล์เก่าหรือเอกสารอ้างอิง (local)
```

## กล้องกรุงธน v2: 112 + สัญญาณไฟ 147/156

โปรไฟล์ `krung_thon_bridge` ใช้พิกัดที่คลิกจากเฟรม 800×450 โดยอัตโนมัติ:

- ROI รวมถนนกล้อง 112
- ROI แยก 4 เลน (`lane_1` ถึง `lane_4`)
- ROI ป้ายไฟหลัก 4 เลนของกล้อง 147 และ 156 พร้อม ROI สำรอง: กล้อง 147
  มีสำรองครบทุกเลน ส่วนกล้อง 156 มีสำรองของไฟลำดับที่ 2 จากซ้าย
  (เทียบเป็นเลน 3 ของกล้อง 112)
- กล้อง 147: `green → up`, `red → down`
- กล้อง 156 อยู่คนละฝั่งและเรียงเลนกลับด้าน: `red → up`, `green → down`
- ถ้ากล้องหนึ่งอ่าน `unknown` จะใช้ผลของอีกกล้องสำหรับเลนนั้น
- ถ้าทั้งคู่ระบุทิศทางตรงกัน จะบันทึกว่า `both`; ถ้าขัดกันจะระบุ `conflict_147_priority`

คำสั่งรันจะจับคู่ไฟล์ `cam112` กับ `cam147` และ `cam156` ในโฟลเดอร์เดียวกันให้อัตโนมัติ:

```bash
python3 vehicle_tracking.py \
  --source locations/krung_thon_bridge/v2/krung_thon_bridge_cam112_v2_1min.mp4 \
  --signal-source locations/krung_thon_bridge/v2/krung_thon_bridge_cam147_v2_1min.mp4 \
  --signal156-source locations/krung_thon_bridge/v2/krung_thon_bridge_cam156_v2_1min.mp4 \
  --model model/coco/yolo11n.pt \
  --output-dir runs/krung_thon_bridge/v2_three_camera \
  --profile krung_thon_bridge \
  --imgsz 640
```

ถ้าวิดีโอเริ่มไม่พร้อมกัน ให้ปรับ `--signal-offset` (กล้อง 147) หรือ
`--signal156-offset` เป็นวินาทีบวกหรือลบได้ ผลใน JSONL จะมี `signal_147_states`,
`signal_156_states`, `lane_signal_fusion`, `lane_id`, `direction`, `expected_direction`
และ `wrong_way` ต่อรถแต่ละคัน โดยจะมี `signal_*_roi_states` และ
`signal_*_selected_light_source` สำหรับตรวจสอบว่าใช้ไฟหลักหรือสำรอง
ระบบจะสร้างวิดีโออธิบายไฟของกล้อง 147 และ 156 เพิ่มให้
โดยอัตโนมัติด้วย

สำหรับวิดีโอที่รันแบบติดตาม (ไม่ใส่ `--raw`) กล่องรถจะเป็นสีเขียวตามปกติ
และจะเป็นสีแดงเมื่อรถเคลื่อนที่สวนกับทิศ `up`/`down` ที่ได้จากไฟของเลนนั้น
บนวิดีโอแสดงเพียงชื่อรถ ไม่แสดง Track ID; ถ้าไฟหรือทิศรถยังอ่านไม่ชัด จะคงสีเขียว
เพื่อไม่แจ้งเตือนเกินจริง ใช้ `--no-wrong-way-alerts` เมื่อต้องการปิดการแจ้งเตือนนี้

### Raw detection ในเขตสแกน แล้วค่อยแยกเลน

เพิ่ม `--raw` เพื่อใช้ YOLO แบบเฟรมต่อเฟรมโดยไม่สร้าง ByteTrack ID ระบบจะ
ปิดภาพนอก ROI สแกนใหญ่ของกล้อง 112 ก่อนส่งให้โมเดล จากนั้นใช้จุดกึ่งกลางล่าง
ของรถกำหนด `lane_id` ว่าอยู่ `lane_1` ถึง `lane_4` ใด ข้อมูล JSONL มีทั้ง
`detections_by_class` และ `detections_by_lane` หากรถยังไม่ตกในกรอบเลนใด จะอยู่
ใน `outside_lane` แทนการถูกตัดทิ้ง ใช้โมเดลรถไทยได้ดังนี้:

```bash
python3 vehicle_tracking.py \
  --raw \
  --source locations/krung_thon_bridge/v3/krung_thon_bridge_cam112_v3_1min.mp4 \
  --signal-source locations/krung_thon_bridge/v3/krung_thon_bridge_cam147_v3_1min.mp4 \
  --signal156-source locations/krung_thon_bridge/v3/krung_thon_bridge_cam156_v3_1min.mp4 \
  --model model/thai_cars_gpu/weights/best.pt \
  --output-dir runs/krung_thon_bridge/v3_raw_scan_roi \
  --profile krung_thon_bridge \
  --imgsz 640 \
  --conf 0.20
```

ไฟล์หลักจะลงท้าย `_raw.mp4` และข้อมูลต่อเฟรมจะอยู่ใน `_raw_detections.jsonl`
โดยไม่มี `track_id`.

ถ้าสีไฟอ่านไม่ได้จากเฟรมใด ระบบจะให้สถานะ `unknown` และจะไม่ตัดสินว่าเป็นรถสวนทาง
เพื่อหลีกเลี่ยงการแจ้งเตือนผิดจากภาพไฟขนาดเล็กหรือภาพเบลอ

พิกัดทั้งหมดเก็บแยกไว้ที่ `config/krung_thon_bridge_regions.py` เพื่อแก้ไขได้โดยไม่
ต้องแตะ logic YOLO ส่วนกลาง

Dataset ที่ใช้อยู่ตอนนี้คือ `data/taksin_vehicles/external/thai_cars_native/`
โดยคงคลาสรถเดิมจาก Thai-Cars และตัด `human` ออก ดูคำสั่งเทรนและใช้งานต่อได้ที่
[`training/README.md`](training/README.md)

## การรันตามสถานที่

ตากสินใช้โมเดลที่เทรนเองและกรอบสะพานเดิม:

```bash
python3 vehicle_tracking.py \
  --model model/taksin/yolo11n_native/weights/best.pt \
  --source archive/locations/taksin/video/taksin_bridge_sathorn_1min.mp4 \
  --output-dir archive/runs/taksin/video/custom_model \
  --merge-car-like --no-show-gates
```

`--merge-car-like` รวม `taxi`, `van` และ `pickup` เป็น `car` ในผลลัพธ์ โดยไม่ต้องเทรนใหม่

ประชานุกูลใช้ YOLO มาตรฐาน COCO เฉพาะ 4 คลาส และปิดกรอบสะพานของตากสิน เพราะ
มุมกล้องต่างกัน:

```bash
python3 vehicle_tracking.py \
  --model model/coco/yolo11m.pt \
  --source archive/locations/prachanukul/video/prachanukul_ratchavipha_1min_timestamp_font16.mp4 \
  --output-dir archive/runs/prachanukul/video/coco_yolo11m \
  --classes car,bus,truck,motorcycle \
  --no-bridge-only --no-show-gates
```
