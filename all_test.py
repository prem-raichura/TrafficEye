import cv2
import numpy as np
import pandas as pd
import time
import os
from ultralytics import YOLO

# === CONFIGURATION ===
SOURCE_MODE = "screen"  # Options: "video", "screen", "image"
FRAME_SKIP = 1
RESIZE_WIDTH = 640

VIDEO_PATH = r"C:\Users\PREM\Desktop\Yolotest\traffic.mp4"
IMAGE_PATH = r"C:\Users\PREM\Desktop\Yolotest\traffic.png"
MODEL_PATH = r"C:\Users\PREM\Desktop\Yolotest\TrafficEye_yolov5su.pt"
#MODEL_PATH = r"C:\Users\PREM\Desktop\Yolotest\TrafficEye_yolov8su.pt"

# === Load YOLO Model ===
model = YOLO(MODEL_PATH)

# === Target Classes ===
VEHICLE_CLASSES = ["Car", "Truck", "Bus", "Motorbike", "Auto", "Person"]

# === DataFrame Logger ===
df = pd.DataFrame(columns=["Timestamp", "Source", "Total Vehicles", "Traffic Density", "FPS"])

# === Traffic Density Logic ===
def calculate_density(count):
    return "High" if count >= 20 else "Medium" if count >= 7 else "Low"

# === Screen Area Selector ===
def get_selected_screen_area():
    import pyautogui
    print("Move mouse to TOP-LEFT corner and press ENTER.")
    input()
    x1, y1 = pyautogui.position()
    print("Move mouse to BOTTOM-RIGHT corner and press ENTER.")
    input()
    x2, y2 = pyautogui.position()
    return {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}

# === Source Initialization ===
if SOURCE_MODE == "screen":
    import mss
    monitor = get_selected_screen_area()
    sct = mss.mss()
elif SOURCE_MODE == "video":
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
elif SOURCE_MODE == "image":
    if not os.path.exists(IMAGE_PATH):
        print("Error: Image not found.")
        exit()
else:
    raise ValueError("Invalid SOURCE_MODE.")

# === Processing Loop ===
frame_count = 0

while True:
    start_time = time.time()

    # === Read Frame ===
    if SOURCE_MODE == "screen":
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame_name = "Screen Capture"
    elif SOURCE_MODE == "video":
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        frame_name = f"Frame {frame_count}"
    elif SOURCE_MODE == "image":
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print("Warning: Image not readable.")
            continue
        frame_name = os.path.basename(IMAGE_PATH)

    # === Frame Skip ===
    if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
        frame_count += 1
        continue

    # === Resize ===
    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        new_h = int(RESIZE_WIDTH / (w / h))
        frame = cv2.resize(frame, (RESIZE_WIDTH, new_h))

    # === YOLOv5 Inference ===
    results = model(frame)
    total_vehicles = 0
    detected_boxes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls_idx = int(box.cls[0].item())
            cls_name = VEHICLE_CLASSES[cls_idx] if cls_idx < len(VEHICLE_CLASSES) else "Unknown"

            if cls_name in VEHICLE_CLASSES and conf > 0.3:
                total_vehicles += 1
                detected_boxes.append((x1, y1, x2, y2, cls_name, conf))

    # === Density & FPS ===
    density = calculate_density(total_vehicles)
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)

    # === Draw Detections ===
    for (x1, y1, x2, y2, cls_name, conf) in detected_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (93, 240, 19), 1)

    # === Create White Space for Stats ===
    stats_height = 50
    stats_canvas = np.ones((stats_height, frame.shape[1], 3), dtype=np.uint8) * 255  # White canvas

    # === Put Stats on the White Canvas ===
    cv2.putText(stats_canvas, f"Total Vehicles: {total_vehicles}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.putText(stats_canvas, f"Traffic Density: {density}", (220, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    
    if SOURCE_MODE != "image":
        cv2.putText(stats_canvas, f"FPS: {int(fps)}", (450, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # === Append Stats Canvas Below Frame ===
    frame_with_stats = np.vstack((frame, stats_canvas))

    # === Log Entry ===
    df.loc[len(df)] = [time.strftime("%Y-%m-%d %H:%M:%S"), frame_name, total_vehicles, density, int(fps)]

    # === Display ===
    cv2.imshow("Traffic Detection", frame_with_stats)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

    # === Image Wait (Only for static image) ===
    if SOURCE_MODE == "image":
        time.sleep(1)

# === Cleanup ===
if SOURCE_MODE == "video":
    cap.release()
cv2.destroyAllWindows()

# === Save CSV ===
df.to_csv("vehicle_density_log.csv", index=False)
print("Detection log saved to vehicle_density_log.csv")

