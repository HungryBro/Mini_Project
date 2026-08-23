# ชุดเทรนรถบนสะพานตากสิน

## ใช้คลาสเดิมจาก Thai-Cars

ชุด `data/Thai-Cars.v2i.yolov11` มีคลาสเดิม 9 คลาส (`bus`, `car`, `human`,
`motorbike`, `pickup`, `taxi`, `truck`, `truck trailer`, `van`) และเป็น polygon
segmentation จึงต้องแปลงเป็นกรอบก่อนใช้กับตัวติดตามปัจจุบัน:

```bash
python training/convert_thai_cars_native.py
```

ผลลัพธ์อยู่ที่ `data/taksin_vehicles/external/thai_cars_native/` โดยไม่รวมคลาส
และไม่ลบ `human` ออก ส่วน polygon จะถูกเปลี่ยนเป็นกรอบสี่เหลี่ยมเท่านั้น

เริ่มเทรนแบบคงคลาสเดิม:

```bash
python training/train.py \
  --data data/taksin_vehicles/external/thai_cars_native/data.yaml \
  --weights model/yolo11n.pt --epochs 50 --imgsz 640 --device cpu \
  --project runs/thai_cars_native --name yolo11n_native
```

เมื่อนำโมเดลไปติดตาม ให้ระบุชื่อคลาสตาม dataset:

```bash
python vehicle_tracking.py --model runs/thai_cars_native/yolo11n_native/weights/best.pt \
  --source taksin_bridge_sathorn/video/taksin_bridge_sathorn_1min.mp4 \
  --classes 'car,truck,bus,van,motorbike,taxi,pickup,human,truck trailer'
```

ปัจจุบันใช้คลาสจาก Thai-Cars เดิมทั้งหมด ไม่ได้รวม `pickup`, `taxi` หรือ `van`
เข้ากับ `car` และไม่ได้ตัด `human` ออก

## ลำดับทำงาน

1. หากต้องสร้างชุดข้อมูลจาก export ใหม่ ให้นำเข้าและตรวจ label ให้เสร็จก่อน
   ส่วนตัวแปลงรุ่นเก่าที่รวมคลาสเหลือ 4 กลุ่มถูกย้ายไป `archive/unused/`
   และไม่ใช่ pipeline ปัจจุบัน

2. สร้างภาพสำหรับติดป้ายกำกับ:

   ```bash
   python training/prepare_dataset.py
   ```

   คำสั่งนี้เก็บภาพทุก 3 วินาที สูงสุด 200 ภาพต่อวิดีโอ พร้อม `manifest.csv` บอกเวลาและวิดีโอต้นทาง

3. สร้างกรอบล่วงหน้าจาก `yolo11m.pt`:

   ```bash
   python training/prelabel_dataset.py --device mps
   ```

   ผลอยู่ใน `data/taksin_vehicles/review/` และยัง **ไม่ใช่** ป้ายกำกับที่เชื่อถือได้

   หากต้องการดูกรอบที่วาดบนภาพ ให้สร้าง preview:

   ```bash
   python training/render_review_previews.py
   ```

   ภาพ preview จะอยู่ใน `data/taksin_vehicles/review/previews/`; ไฟล์ภาพต้นฉบับและ label จะไม่ถูกแก้ไข

4. เปิดภาพและไฟล์ `.txt` ในเครื่องมือติดป้ายกำกับที่รองรับ YOLO แล้วตรวจแก้ทุกภาพ โดยเฉพาะรถบัส/รถบรรทุก, รถไกล, มอเตอร์ไซค์ และรถที่อยู่นอกตัวสะพาน

   - ลบกรอบนอกสะพาน, BTS, ถนนข้างทาง และสิ่งที่ไม่ใช่รถ
   - เพิ่มรถที่ pre-label พลาด
   - เปลี่ยนบัสที่ถูกติดเป็น truck และกลับกัน
   - ภาพที่ไม่มีรถบนสะพานต้องคงไฟล์ label ว่างไว้

5. ตรวจความถูกต้องและแบ่ง train/validation/test ตามช่วงเวลา:

   ```bash
   python training/split_reviewed_dataset.py
   ```

6. เทรนชุด 9 คลาสปัจจุบัน:

   ```bash
   python training/train.py --device mps
   ```

   โมเดลที่ได้อยู่ใต้ `runs/taksin_training/yolo11m_native/weights/best.pt` แล้วนำไปใช้กับ `vehicle_tracking.py` ด้วย `--model` ได้

## ข้อควรระวัง

- เริ่มเทรนเมื่อมีภาพตรวจแก้อย่างน้อย 300 ภาพ และให้มี bus/truck อย่างละประมาณ 100 คันขึ้นไป
- อย่าเทรนจากผล pre-label โดยไม่ตรวจ เพราะจะตอกย้ำความผิดพลาด bus/truck เดิม
- หากมี checkpoint ของ Thailand-vehicles ที่ export เป็น YOLO `.pt` แล้ว สามารถใช้แทนจุดตั้งต้นได้: `python training/train.py --weights path/to/thailand_vehicles.pt`
