# EXPERIMENT: run the RF-DETR detector on out-of-distribution Monza F1 footage. RESULT: degraded performance off-domain, as expected.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
Racetrack Border Detection - Video Inference
Testing out-of-distribution (Monza F1 video) — expect degraded performance
"""

from roboflow import Roboflow
import cv2
import numpy as np
import os
import tempfile

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ.get("ROBOFLOW_API_KEY")  # was a hardcoded key; rotated and redacted
WORKSPACE = "revazs-workspace"
PROJECT = "racetrack-border-detection-off"
VERSION = 1

INPUT_VIDEO = r"C:\Users\haise\Downloads\hamilton_monza_2020.mp4"   # <-- rename if needed
OUTPUT_VIDEO = r"C:\Users\haise\Documents\Camera_line\detecting_from_random.mp4"

FRAME_SKIP = 10                 # bigger skip — it's a fast lap, 1:18 at 30fps = ~2400 frames
CONFIDENCE = 30                 # lower threshold to catch any weak predictions
MAX_POLYGON_AREA_RATIO = 0.35
DRAW_FILL = False

COLORS = {
    "painted_line": (0, 255, 255),   # yellow
    "edge": (0, 165, 255),           # orange
}
DEFAULT_COLOR = (0, 255, 0)

# ============================================================
# SETUP
# ============================================================

print("Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)

try:
    project = rf.workspace(WORKSPACE).project(PROJECT)
except Exception:
    project = rf.workspace().project(PROJECT)

model = project.version(VERSION).model
print(f"Loaded model: {PROJECT} v{VERSION}")
print("NOTE: This model was trained on Yas Marina.")
print("Running on Monza F1 footage — expect out-of-distribution performance.\n")

# ============================================================
# OPEN INPUT
# ============================================================

if not os.path.exists(INPUT_VIDEO):
    raise RuntimeError(f"Video not found: {INPUT_VIDEO}")

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS)
output_fps = max(1, src_fps / FRAME_SKIP)

print(f"Input: {INPUT_VIDEO}")
print(f"  Frames: {total_frames}")
print(f"  Size:   {w}x{h}")
print(f"  FPS:    {src_fps:.1f}")
print(f"Output FPS: {output_fps:.1f}\n")

# ============================================================
# OUTPUT
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, output_fps, (w, h))
if not out.isOpened():
    raise RuntimeError(f"Could not open video writer for {OUTPUT_VIDEO}")

# ============================================================
# PROCESS
# ============================================================

temp_dir = tempfile.mkdtemp()
temp_frame_path = os.path.join(temp_dir, "frame.jpg")

seen_classes = set()
rejected_large = 0
accepted = 0
frame_idx = 0
processed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP != 0:
        frame_idx += 1
        continue

    processed += 1
    print(f"[{frame_idx}/{total_frames}]", end=" ")

    cv2.imwrite(temp_frame_path, frame)

    try:
        result = model.predict(temp_frame_path, confidence=CONFIDENCE)
        predictions = result.json().get("predictions", [])
        print(f"{len(predictions)} preds")

        for pred in predictions:
            cls = pred.get("class", "unknown")
            conf = pred.get("confidence", 0)
            seen_classes.add(cls)

            raw_points = pred.get("points", [])
            if len(raw_points) < 3:
                continue

            pts = np.array(
                [[int(p["x"]), int(p["y"])] for p in raw_points],
                dtype=np.int32,
            )

            bbox_w = pts[:, 0].max() - pts[:, 0].min()
            bbox_h = pts[:, 1].max() - pts[:, 1].min()
            area_ratio = (bbox_w * bbox_h) / (w * h)

            if area_ratio > MAX_POLYGON_AREA_RATIO:
                rejected_large += 1
                continue

            accepted += 1
            color = COLORS.get(cls, DEFAULT_COLOR)

            if DRAW_FILL:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)

            top_idx = np.argmin(pts[:, 1])
            label_pos = (int(pts[top_idx, 0]), int(pts[top_idx, 1]) - 5)
            cv2.putText(
                frame, f"{cls} {conf:.0%}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

    except Exception as e:
        print(f"ERROR: {e}")

    # Header: warning + legend
    cv2.rectangle(frame, (0, 0), (360, 110), (0, 0, 0), -1)
    cv2.putText(frame, "Out-of-distribution test", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
    cv2.putText(frame, "Model: Yas Marina", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, (cname, cval) in enumerate(COLORS.items()):
        cv2.putText(frame, cname, (10, 80 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cval, 2)

    cv2.putText(frame, f"frame {frame_idx}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

try:
    os.remove(temp_frame_path)
    os.rmdir(temp_dir)
except Exception:
    pass

print(f"\n{'=' * 50}")
print(f"Classes seen:     {sorted(seen_classes)}")
print(f"Processed:        {processed} frames")
print(f"Accepted preds:   {accepted}")
print(f"Rejected (large): {rejected_large}")
print(f"Output:           {OUTPUT_VIDEO}")
print(f"{'=' * 50}")