import cv2

# โหลดภาพเฟรมแรกจากวิดีโอ
cap = cv2.VideoCapture('/Users/dolphin/Desktop/Mini Project/locations/krung_thon_bridge/v2_have_wrrongway/krung_thon_bridge_cam112_v2_1min.mp4')
ret, frame = cap.read()
cap.release()

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"ปักจุดที่ {len(points)}: [{x}, {y}]")
        # วาดจุดและเส้นบนภาพ
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 255), 2)
        cv2.imshow("Click to select Polygon points", frame)

if ret:
    print("=== วิธีใช้งาน ===")
    print("1. คลิกเมาส์ซ้ายล้อมรอบพื้นที่เกาะสีขาว (เรียงตามลำดับ วงรอบเกาะ)")
    print("2. เมื่อคลิกครบแล้ว กดปุ่ม 'q' บนคีย์บอร์ด เพื่อปริ้นท์พิกัดไปใช้งาน")
    cv2.imshow("Click to select Polygon points", frame)
    cv2.setMouseCallback("Click to select Polygon points", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n--- คัดลอกพิกัดนี้ไปใส่ใน POLYGON_ZONE ---")
    print(f"POLYGON_ZONE = np.array({points}, np.int32)")