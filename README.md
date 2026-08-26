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
- ไฟที่ชัดจากกล้องใดกล้องหนึ่งใช้ได้ แต่สองกล้องที่เห็นทิศเดียวกันจะมี
  confidence สูงกว่า
- ผลรวมระบุแหล่งตัดสินเป็น `visual_both`, `visual_147`, `visual_156`,
  `schedule_fallback` หรือ `schedule_conflict_safe`

เพิ่มกติกาความมั่นใจสำหรับภาพไฟที่เบลอหรือสะท้อนดังนี้:

- ROI ไฟแต่ละจุดคำนวณ confidence จากสัดส่วนสีและการแยกสีแดง/เขียว แล้วใช้
  ประวัติแบบถ่วงน้ำหนัก 15 เฟรมเพื่อลดการสลับเพียงเฟรมเดียว
- ระบบอ่าน timestamp เฉพาะเฟรมแรกของกล้อง 112, 147 และ 156 แล้วใช้เวลาเดินของ
  วิดีโอแต่ละกล้องเองเป็นเสียงโหวตตารางเวลา สามกล้องเห็นตารางเดียวกัน = `0.92`,
  สองกล้อง = `0.85`, หนึ่งกล้อง = `0.75` สามารถกำหนดเวลาเองด้วย
  `--timestamp-112`, `--timestamp-147`, `--timestamp-156` หาก OCR อ่านตัวเลขไม่ครบ
- ตารางรายวัน: 05:30-07:30 และ 08:20-08:45 เป็น `up/up/up/down`;
  15:30-16:30, 17:00-17:45 และ 18:15-20:30 เป็น `up/down/down/down`;
  นอกนั้นเป็น `up/up/down/down`
- เมื่อไฟที่ชัดเจนขัดกับตาราง ระบบจะแสดง `schedule_conflict_safe` แต่หยุด
  แจ้งรถย้อนศรของเลนนั้นชั่วคราวเพื่อไม่ให้กล่องแดงผิดพลาด หากไฟอ่อน, unknown
  หรือสองกล้องขัดกัน จะใช้ `schedule_fallback` แทน

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
ในโปรไฟล์กรุงธน ชื่อแสดงผลของ COCO `motorcycle` จะย่อเป็น `moto`
โดยไม่เปลี่ยน class ID ภายในโมเดล

สำหรับวิดีโอที่รันแบบติดตาม (ไม่ใส่ `--raw`) กล่องรถจะเป็นสีเขียวตามปกติ
และจะเป็นสีแดงเมื่อรถเคลื่อนที่สวนกับทิศ `up`/`down` ที่ยืนยันได้จากไฟ
หรือ `schedule_fallback` ของเลนนั้น บนวิดีโอแสดงเพียงชื่อรถ ไม่แสดง Track ID;
กรณี `unknown` หรือ `schedule_conflict_safe` จะคงสีเขียวเพื่อไม่แจ้งเตือนเกินจริง
ใช้ `--no-wrong-way-alerts` เมื่อต้องการปิดการแจ้งเตือนนี้
เส้นกรอบเลน ข้อความเลน และลูกศรทิศทางบนกล้อง 112 เป็นสีขาวเสมอ เพื่อไม่ปะปนกับ
สีเขียว/แดงของกล่องรถ

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

ถ้าสีไฟอ่านไม่ได้จากเฟรมใด ระบบจะให้สถานะ `unknown` แล้วใช้ตารางเวลา
เฉพาะเมื่อ timestamp ของกล้องยังอ่านได้ หากไฟที่ชัดเจนขัดกับตารางหรือไม่มี
timestamp ที่ใช้ได้ จะไม่ตัดสินว่าเป็นรถสวนทางเพื่อหลีกเลี่ยงการแจ้งเตือนผิด

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
