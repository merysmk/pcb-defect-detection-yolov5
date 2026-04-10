# ===================== FULL SCRIPT: Stable YOLOv5 + Auto Export + HTML Report + LATEST JSON =====================
# Requirements:
#   pip install opencv-python torch numpy
#trained weights: best.pt

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import os
import json
import datetime
import time
from collections import deque, Counter

import cv2
import torch
import numpy as np


# --------------------------- CONFIG ---------------------------
WEIGHTS_PATH = "best.pt"

# YOLO thresholds
YOLO_CONF = 0.5
YOLO_IOU = 0.5
INFER_SIZE = 640

# Camera
CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720

# Tracking / stability
IOU_THRESH = 0.35
CONFIRM_STREAK = 4      # frames CONSÉCUTIVES to confirm
MAX_MISS = 8            # remove track after this many missed frames
HOLD_TIME = 0.35        # seconds to keep drawing after losing a detection briefly
ALPHA_BOX = 0.35        # EMA smoothing for bbox
ALPHA_CONF = 0.25       # EMA smoothing for confidence

# Export
OUTPUT_ROOT = "inspections"
LATEST_JSON_NAME = "PCB_latest.json"   # Unity reads this always
LATEST_IMG_NAME = "PCB_latest.jpg"     # Unity reads this always (latest annotated image)
WINDOW_NAME = "Stable Detection"
# --------------------------------------------------------------


# --------------------------- LOAD MODEL ---------------------------
model = torch.hub.load("ultralytics/yolov5", "custom", path=WEIGHTS_PATH, force_reload=False)
model.conf = YOLO_CONF
model.iou = YOLO_IOU
# --------------------------------------------------------------


# --------------------------- IOU ---------------------------
def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(inter / (areaA + areaB - inter + 1e-9))
# --------------------------------------------------------------


# --------------------------- TRACK CLASS ---------------------------
class Track:
    def __init__(self, box, label, conf, now):
        self.box = np.array(box, dtype=float)
        self.label_hist = deque([label], maxlen=8)
        self.conf_ema = float(conf)

        self.miss = 0
        self.streak = 1          # consecutive hits
        self.confirmed = False

        self.last_seen = now
        self.hold_until = 0.0    # keep drawing until this time even if briefly missed

    @property
    def label(self):
        return Counter(self.label_hist).most_common(1)[0][0]
# --------------------------------------------------------------


# --------------------------- RISK/ACTION RULES ---------------------------
def rule_risk_action(label: str):
    """
    Modify these rules to match your dataset labels.
    """
    l = label.lower()

    if "short" in l:
        return "HIGH", "REJECT"
    if "missing" in l and ("component" in l or "part" in l):
        return "HIGH", "REWORK"
    if "missing" in l and ("hole" in l or "via" in l):
        return "MEDIUM", "REWORK"
    if "scratch" in l or "spurious" in l:
        return "LOW", "OK"

    return "MEDIUM", "REWORK"


def decide_board(defects):
    if not defects:
        return "OK"
    risks = [d["risk"] for d in defects]
    actions = [d["action"] for d in defects]
    if "HIGH" in risks or "REJECT" in actions:
        return "REJECT"
    return "REWORK"
# --------------------------------------------------------------


# --------------------------- EXPORT ---------------------------
def export_inspection(output_root, frame_bgr, confirmed_tracks):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(output_root, f"inspection_{ts}")
    os.makedirs(folder, exist_ok=True)

    # Save final frame
    img_path = os.path.join(folder, "frame_final.jpg")
    cv2.imwrite(img_path, frame_bgr)

    defects = []
    for tr in confirmed_tracks:
        label = tr.label
        conf = float(tr.conf_ema)
        x1, y1, x2, y2 = [int(v) for v in tr.box.tolist()]
        risk, action = rule_risk_action(label)

        defects.append({
            "label": label,
            "confidence": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2],
            "risk": risk,
            "action": action
        })

    # Stats
    by_type = {}
    for d in defects:
        by_type[d["label"]] = by_type.get(d["label"], 0) + 1

    mean_conf = round(sum(d["confidence"] for d in defects) / max(1, len(defects)), 4)
    decision = decide_board(defects)

    payload = {
        "timestamp": ts,
        "total_defects": len(defects),
        "defects_by_type": by_type,
        "mean_confidence": mean_conf,
        "final_decision": decision,
        "defects": defects
    }

    # Save historical JSON
    json_path = os.path.join(folder, "defects.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Save HTML report
    html_path = os.path.join(folder, "report.html")
    rows = ""
    for d in defects:
        rows += f"""
        <tr>
          <td>{d['label']}</td>
          <td>{d['confidence']}</td>
          <td>{d['risk']}</td>
          <td>{d['action']}</td>
          <td>{d['bbox_xyxy']}</td>
        </tr>
        """

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PCB Inspection Report</title>
  <style>
    body {{ font-family: Arial; margin: 20px; }}
    .badge {{ display:inline-block; padding:6px 10px; border-radius:8px; background:#eee; }}
    table {{ border-collapse: collapse; width:100%; margin-top: 10px; }}
    th, td {{ border:1px solid #ddd; padding: 8px; text-align:left; }}
    th {{ background:#f6f6f6; }}
  </style>
</head>
<body>
  <h2>PCB Inspection Report</h2>
  <div><b>Timestamp:</b> {ts}</div>
  <div><b>Total defects:</b> {len(defects)}</div>
  <div><b>Mean confidence:</b> {mean_conf}</div>
  <div><b>Final decision:</b> <span class="badge">{decision}</span></div>

  <h3>Defects</h3>
  <table>
    <tr><th>Label</th><th>Conf</th><th>Risk</th><th>Action</th><th>BBox</th></tr>
    {rows if rows else "<tr><td colspan='5'>No defects</td></tr>"}
  </table>

  <p><b>Image saved:</b> frame_final.jpg</p>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return folder, payload
# --------------------------------------------------------------


# --------------------------- MAIN ---------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    tracks = []
    last_frame = None

    while True:
        now = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()

        # YOLO detect
        results = model(frame, size=INFER_SIZE)
        det = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

        # mark all tracks missed at start
        for tr in tracks:
            tr.miss += 1

        used_tracks = set()

        # associate each detection to a track
        for x1, y1, x2, y2, conf, cls in det:
            box = np.array([x1, y1, x2, y2], dtype=float)
            label = results.names[int(cls)]
            conf = float(conf)

            best_iou = 0.0
            best_idx = -1

            for idx, tr in enumerate(tracks):
                if idx in used_tracks:
                    continue
                i = iou(tr.box, box)
                if i > best_iou:
                    best_iou = i
                    best_idx = idx

            if best_idx != -1 and best_iou >= IOU_THRESH:
                tr = tracks[best_idx]
                used_tracks.add(best_idx)

                tr.box = ALPHA_BOX * box + (1 - ALPHA_BOX) * tr.box
                tr.conf_ema = ALPHA_CONF * conf + (1 - ALPHA_CONF) * tr.conf_ema
                tr.label_hist.append(label)

                tr.miss = 0
                tr.last_seen = now
                tr.streak += 1

                if (not tr.confirmed) and tr.streak >= CONFIRM_STREAK:
                    tr.confirmed = True

                tr.hold_until = now + HOLD_TIME
            else:
                tracks.append(Track(box, label, conf, now))

        # tracks not seen this frame: reset streak
        for tr in tracks:
            if tr.miss > 0:
                tr.streak = 0

        # remove old tracks
        tracks = [t for t in tracks if t.miss <= MAX_MISS]

        # draw confirmed tracks (with hold)
        display = frame.copy()
        for tr in tracks:
            show = tr.confirmed and (tr.miss == 0 or now <= tr.hold_until)
            if not show:
                continue

            x1, y1, x2, y2 = tr.box.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{tr.label} {tr.conf_ema:.2f}"
            cv2.putText(display, txt, (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            confirmed = [tr for tr in tracks if tr.confirmed]
            if last_frame is None:
                last_frame = display.copy()

            #  CHANGEMENT UNIQUE: exporter l'image annotée (display), pas l'image brute (last_frame)
            folder, payload = export_inspection(OUTPUT_ROOT, display, confirmed)

            # NEW: Update LATEST IMAGE for Unity (fixed file)
            latest_img_path = os.path.join(OUTPUT_ROOT, LATEST_IMG_NAME)
            cv2.imwrite(latest_img_path, display)

            # Update LATEST JSON for Unity (fixed file)
            latest_path = os.path.join(OUTPUT_ROOT, LATEST_JSON_NAME)
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            print(f"[OK] Exported: {folder}")
            print(f"[OK] Latest JSON updated: {latest_path}")
            print(f"[OK] Latest image updated: {latest_img_path}")
            print(f"Decision: {payload['final_decision']} | Defects: {payload['total_defects']}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===================== END =====================
