"""แก้เฉพาะไฟล์นี้เพื่อเลือกว่า live view จะใช้คลิปใดของกล้อง 112, 147 และ 156.

หลังแก้พาธแล้ว เปิด Terminal ในโฟลเดอร์ tracking และรัน:
    python3 vehicle_tracking.py

กด q ขณะหน้าต่างวิดีโอเปิดอยู่เพื่อหยุดทันที. ระบบจะไม่บันทึก MP4 แต่จะ
เขียน JSONL ต่อเนื่องลงใน LOG_DIRECTORY ระหว่างที่กำลังแสดงผล.
"""

from pathlib import Path


TRACKING_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = TRACKING_DIRECTORY.parent

# เลือกคลิปของแต่ละกล้องตรงนี้ได้เลย. ค่าเริ่มต้นเป็นชุด 10 นาทีล่าสุด.
CAMERA_112_SOURCE = PROJECT_DIRECTORY / "locations/krung_thon_bridge/v7/raw_timestamped/krung_thon_bridge_cam112_v7_raw_timestamped_10min.mp4"
CAMERA_147_SOURCE = PROJECT_DIRECTORY / "locations/krung_thon_bridge/v7/raw_timestamped/krung_thon_bridge_cam147_v7_raw_timestamped_10min.mp4"
CAMERA_156_SOURCE = PROJECT_DIRECTORY / "locations/krung_thon_bridge/v7/raw_timestamped/krung_thon_bridge_cam156_v7_raw_timestamped_10min.mp4"

# โมเดลและตัวเลือกตรวจจับ
# สามารถเลือกใช้: yolo11n.pt (เร็วลื่นที่สุด ~60FPS), yolo11s.pt (เร็วและแม่นยำ), yolo11m.pt (หนัก)
MODEL_PATH = PROJECT_DIRECTORY / "model/coco/yolo11n.pt"
CONFIDENCE = 0.25
IOU = 0.30
IMAGE_SIZE = 640
DEVICE = "mps"  # ใช้ Apple Silicon GPU Acceleration ให้ทำงานลื่นไหลเหมือนดูวิดีโอปกติ
TRACKER_PATH = PROJECT_DIRECTORY / "config/vehicle_bytetrack.yaml"

# การแสดงผล/ตรรกะของสะพานกรุงธน
PROFILE = "krung_thon_bridge"
BRIDGE_ONLY = True
SHOW_LANES = True
SHOW_GATES = False
AGNOSTIC_NMS = True
WRONG_WAY_ALERTS = True
OCCLUSION_HOLD = 8
MAX_HELD_TRACKS = 3
SIGNAL_147_OFFSET_SECONDS = 0.0
SIGNAL_156_OFFSET_SECONDS = 0.0

# ใช้ None เพื่อให้อ่านเวลา OSD บนคลิปอัตโนมัติ. หาก OCR อ่านไม่ได้ สามารถ
# กำหนดสตริง ISO (เช่น "2026-08-26 07:47:28") ได้ที่นี่
TIMESTAMP_112 = None
TIMESTAMP_147 = None
TIMESTAMP_156 = None

# ไดเรกทอรีสำหรับบันทึกไฟล์ JSONL log
LOG_DIRECTORY = PROJECT_DIRECTORY / "runs/live_logs"

# โหมดการส่งข้อมูล Payload ออกไปยังระบบภายนอก
# เลือกระบุโปรโตคอลที่ต้องการรัน: "udp", "mqtt", "both", หรือ "none"
PAYLOAD_PROTOCOL = "mqtt"  # เลือกสลับระหว่าง "udp" หรือ "mqtt" ตรงนี้ได้เลย

WINDOW_SECONDS = 30.0
USE_WALL_CLOCK_TIME = True

# ตั้งค่าสำหรับ UDP
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005

# ตั้งค่าสำหรับ MQTT
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "traffic/krung_thon_bridge/summary"

# ตัวแปรระบบเปิด/ปิดการทำงานอัตโนมัติตาม PAYLOAD_PROTOCOL
ENABLE_UDP_PAYLOAD = PAYLOAD_PROTOCOL.lower() in ("udp", "both")
ENABLE_MQTT_PAYLOAD = PAYLOAD_PROTOCOL.lower() in ("mqtt", "both")
