import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # Fix Windows path

import torch
import cv2
import time
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes

# === PARAMS ===
MODEL_PATH = r"C:/Users/pc/Desktop/ESP32-CAM_IA_AR/best.pt"
ESP32_URL = "http://10.161.158.181:81/stream"
CONF_THRESHOLD = 0.25
IMG_SIZE = 416
FRAME_SKIP = 3
DEVICE = "cpu"
DETECTION_KEEP_FRAMES = 5  # Combien de frames garder la détection

# === LOAD MODEL ===
model = DetectMultiBackend(MODEL_PATH, device=DEVICE)
model.conf = CONF_THRESHOLD

# === CAMERA ===
cap = cv2.VideoCapture(ESP32_URL)
if not cap.isOpened():
    print("❌ Impossible d'accéder au flux")
    exit()

frame_count = 0
start_time = time.time()
recent_detections = []  # stocke les dernières detections
recent_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    frame_count += 1
    frame_out = frame.copy()

    # === DETECTION YOLO toutes les FRAME_SKIP frames ===
    if frame_count % FRAME_SKIP == 0:
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            pred = model(img)
        pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD)[0]

        # Stocker les detections pour les garder plusieurs frames
        recent_detections = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred:
                xyxy = [int(x.item()) for x in xyxy]
                recent_detections.append((xyxy, float(conf.item()), int(cls)))
        recent_count = DETECTION_KEEP_FRAMES

    # === AFFICHAGE DES DETECTIONS MEMORISEES ===
    if recent_count > 0:
        for xyxy, conf, cls_id in recent_detections:
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame_out, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame_out, label, (xyxy[0], xyxy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        recent_count -= 1

    # === FPS ===
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame_out, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # === AFFICHAGE ===
    cv2.imshow("YOLOv5 PCB Defects", frame_out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()