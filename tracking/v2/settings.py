"""ตั้งค่า live view v2 ที่ใช้กล้อง 112 เพียงกล้องเดียว.

หลังแก้พาธแล้ว เปิด Terminal ในโฟลเดอร์ tracking/v2 และรัน:
    cd "/Users/dolphin/Desktop/Mini Project/tracking/v2"
    python3 vehicle_tracking.py

กด q ขณะหน้าต่างวิดีโอเปิดอยู่เพื่อหยุดทันที. เมื่อใช้ live_stream ระบบจะ
อัดภาพที่แสดงผลเป็น MP4 และเขียน JSONL ต่อเนื่องลงใน LOG_DIRECTORY
ระหว่างที่กำลังแสดงผล.
"""

from pathlib import Path


TRACKING_DIRECTORY = Path(__file__).resolve().parent
# v2 อยู่ใน tracking/v2 จึงย้อนขึ้นสองระดับเพื่อหารากของโปรเจกต์
PROJECT_DIRECTORY = TRACKING_DIRECTORY.parents[1]

# โหมดแหล่งข้อมูลวิดีโอ ("video_files" หรือ "live_stream")
SOURCE_MODE = "live_stream"  # ใช้ "video_files" เมื่อต้องการทดสอบจากคลิปในเครื่อง

# ลิงก์สตรีมสด HLS (.m3u8) ของกล้อง 112
CAMERA_112_STREAM_URL = "https://drr-kt-svr02.enixma.net/live/192.168.8.112.stream/playlist.m3u8"

# คลิปวิดีโอไฟล์ในเครื่องสำหรับทดสอบ
CAMERA_112_FILE = PROJECT_DIRECTORY / "locations/krung_thon_bridge/v2_have_wrrongway/krung_thon_bridge_cam112_v2_1min.mp4"

# สวิตช์สลับแหล่งข้อมูลวิดีโอตาม SOURCE_MODE
if str(SOURCE_MODE).lower() in ("live_stream", "stream", "url", "live"):
    CAMERA_112_SOURCE = CAMERA_112_STREAM_URL
else:
    CAMERA_112_SOURCE = CAMERA_112_FILE

# โมเดลและตัวเลือกตรวจจับ
# สามารถเลือกใช้: yolo11n.pt (เร็วลื่นที่สุด ~60FPS), yolo11s.pt (เร็วและแม่นยำสูง), yolo11m.pt (หนัก)
MODEL_PATH = PROJECT_DIRECTORY / "model/thai_cars_gpu/weights/best.pt"
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

# ID ของ vehicle จะแยกเลขตามเลน เช่น L1-V001 และพยายามเชื่อม ID เดิม
# เมื่อ ByteTrack หลุดเพราะรถบังกันชั่วคราว. ค่าระยะเป็นพิกเซลของภาพกล้อง 112.
VEHICLE_ID_REASSOCIATE_SECONDS = 2.0
VEHICLE_ID_MAX_CENTER_DISTANCE = 70.0
VEHICLE_ID_MIN_IOU = 0.05

# จะยืนยันว่าเป็นรถย้อนศรก็ต่อเมื่อรถเคลื่อนสวนทิศต่อเนื่องตามจำนวนเฟรม
# และมีระยะเคลื่อนที่สะสมพอสมควรแล้ว จึงนับเป็นเหตุการณ์เพียงครั้งเดียว.
WRONG_WAY_CONFIRM_FRAMES = 6
WRONG_WAY_MIN_DISPLACEMENT_PIXELS = 12.0

# ใช้ None เพื่อให้อ่านเวลา OSD บนภาพวิดีโอ/สตรีมให้อัตโนมัติ
# หากวิดีโอไม่มีข้อความเวลา หรือต้องการระบุเวลาเริ่มต้นเอง สามารถใส่สตริง เช่น "2026-08-26 07:47:28" ได้
TIMESTAMP_112 = None

# ถ้า OCR เวลาในกล้อง 112 ไม่ได้ ให้ใช้เวลาปัจจุบันของเครื่อง (เวลาไทย)
# เพื่อเลือกตารางเลนแทน. False = ระบุทิศทางเป็น unknown เพื่อความปลอดภัย.
USE_SYSTEM_CLOCK_IF_112_TIME_UNAVAILABLE = True

# ไดเรกทอรีสำหรับบันทึกแต่ละ live session (MP4 และ JSONL)
LOG_DIRECTORY = PROJECT_DIRECTORY / "runs/live_logs"

# อัดวิดีโอเฉพาะเมื่อ SOURCE_MODE เป็น live_stream. วิดีโอที่อัดคือภาพ
# เดียวกับที่แสดงบนจอ (มีกรอบ/ชื่อวัตถุ/เส้นเลน) เพื่อใช้ตรวจย้อนหลัง.
RECORD_LIVE_MP4 = True
# เลือกอัตโนมัติจาก FPS ของกล้อง: 25 FPS จะบันทึก 25 FPS, 30 FPS จะบันทึก 30 FPS.
LIVE_RECORDING_FPS_FALLBACK = 25.0

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
