"""แก้เฉพาะไฟล์นี้เพื่อเลือกว่า live view จะใช้คลิปใดของกล้อง 112, 147 และ 156.

หลังแก้พาธแล้ว เปิด Terminal ในโฟลเดอร์ tracking/v1 และรัน:
    cd "/Users/dolphin/Desktop/Mini Project/tracking/v1"
    python3 vehicle_tracking.py

กด q ขณะหน้าต่างวิดีโอเปิดอยู่เพื่อหยุดทันที. ระบบจะไม่บันทึก MP4 แต่จะ
เขียน JSONL ต่อเนื่องลงใน LOG_DIRECTORY ระหว่างที่กำลังแสดงผล.
"""

from pathlib import Path


TRACKING_DIRECTORY = Path(__file__).resolve().parent
# v1 อยู่ใน tracking/v1 จึงย้อนขึ้นสองระดับเพื่อหารากของโปรเจกต์
PROJECT_DIRECTORY = TRACKING_DIRECTORY.parents[1]

# โหมดแหล่งข้อมูลวิดีโอ ("video_files" หรือ "live_stream")
SOURCE_MODE = "live_stream"  # เปลี่ยนเป็น "live_stream" เพื่อดึงภาพกล้องสดจากลิงก์ URL , video_files 

# ลิงก์สตรีมสด HLS (.m3u8) ของแต่ละกล้อง
CAMERA_112_STREAM_URL = "https://drr-kt-svr02.enixma.net/live/192.168.8.112.stream/playlist.m3u8"
CAMERA_147_STREAM_URL = "https://drr-kt-svr02.enixma.net/live/192.168.8.147.stream/playlist.m3u8"
CAMERA_156_STREAM_URL = "https://drr-kt-svr02.enixma.net/live/192.168.8.156.stream/playlist.m3u8"

# คลิปวิดีโอไฟล์ในเครื่องสำหรับทดสอบ
CAMERA_112_FILE = PROJECT_DIRECTORY / "locations/krung_thon_bridge/v2_have_wrrongway/krung_thon_bridge_cam112_v2_1min.mp4"
CAMERA_147_FILE = PROJECT_DIRECTORY / "locations/krung_thon_bridge/v2_have_wrrongway/krung_thon_bridge_cam147_v2_1min.mp4"
CAMERA_156_FILE = PROJECT_DIRECTORY / "locations/krung_thon_bridge/v2_have_wrrongway/krung_thon_bridge_cam156_v2_1min.mp4"

# สวิตช์สลับแหล่งข้อมูลวิดีโอตาม SOURCE_MODE
if str(SOURCE_MODE).lower() in ("live_stream", "stream", "url", "live"):
    CAMERA_112_SOURCE = CAMERA_112_STREAM_URL
    CAMERA_147_SOURCE = CAMERA_147_STREAM_URL
    CAMERA_156_SOURCE = CAMERA_156_STREAM_URL
else:
    CAMERA_112_SOURCE = CAMERA_112_FILE
    CAMERA_147_SOURCE = CAMERA_147_FILE
    CAMERA_156_SOURCE = CAMERA_156_FILE

# โมเดลและตัวเลือกตรวจจับ
# สามารถเลือกใช้: yolo11n.pt (เร็วลื่นที่สุด ~60FPS), yolo11s.pt (เร็วและแม่นยำสูง), yolo11m.pt (หนัก)
MODEL_PATH = PROJECT_DIRECTORY / "model/coco/yolo11s.pt"
CONFIDENCE = 0.35
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

# ใช้ None เพื่อให้อ่านเวลา OSD บนภาพวิดีโอ/สตรีมให้อัตโนมัติ
# หากวิดีโอไม่มีข้อความเวลา หรือต้องการระบุเวลาเริ่มต้นเอง สามารถใส่สตริง เช่น "2026-08-26 07:47:28" ได้
TIMESTAMP_112 = None
TIMESTAMP_147 = None
TIMESTAMP_156 = None

# ไดเรกทอรีสำหรับบันทึกไฟล์ JSONL log
LOG_DIRECTORY = PROJECT_DIRECTORY / "runs/live_logs"

# โหมดการส่งข้อมูล Payload ออกไปยังระบบภายนอก
# เลือกระบุโปรโตคอลที่ต้องการรัน: "udp", "mqtt", "both", หรือ "none"
PAYLOAD_PROTOCOL = "mqtt"  # เลือกสลับระหว่าง "udp" หรือ "mqtt" ตรงนี้ได้เลย

WINDOW_SECONDS = 30.0
# สลับใช้ Wall-clock time อัตโนมัติ: True เมื่อรันกล้องสด (live_stream) และ False เมื่อรันไฟล์คลิปวิดีโอ
USE_WALL_CLOCK_TIME = str(SOURCE_MODE).lower() in ("live_stream", "stream", "url", "live")

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
