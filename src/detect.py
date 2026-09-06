"""
Stage 1: 2D racetrack boundary detection with RF-DETR (hosted on Roboflow).

Consolidates the three detection scripts that were previously separate:
  yas_detect.py            single-frame inference + overlay
  detect_borders_video.py  frame-sequence inference, temporal smoothing, video out
  show_detection_2d.py     polygon overlay rendering

Detects two classes: painted_line and track_edge.

The API key is read from the ROBOFLOW_API_KEY environment variable. It is never
stored in this file.

    PowerShell:  $env:ROBOFLOW_API_KEY = "<key>"

Usage:
    # single image -> overlay png
    python src/detect.py --input frame_000050.png --output overlay.png

    # directory of frames -> annotated video (temporal smoothing on)
    python src/detect.py --input camera_frames/ --output detection.mp4

    # directory of frames -> per-class binary masks for the unprojection stage
    python src/detect.py --input camera_frames/ --masks-out masks/
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = "revazs-workspace"
PROJECT = "racetrack-border-detection-off"

# The three source scripts disagreed on model version (1, 2 and 3 respectively).
# Version 2 produced racetrack_border_detection_v2.mp4, so it is the default here.
# Override with --model-version once you confirm which version the reported
# precision/recall figures came from.
DEFAULT_VERSION = 2

# Class colours in BGR. Both the long and short class names are accepted because
# the deployed model versions have used both schemes.
COLORS = {
    "painted_line": (0, 255, 255),
    "line": (0, 255, 255),
    "track_edge": (0, 165, 255),
    "edge": (0, 165, 255),
}
DEFAULT_COLOR = (0, 255, 0)

# Canonical names, used for mask filenames so downstream stages see one scheme.
CANONICAL = {"line": "painted_line", "edge": "track_edge"}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set.\n"
            '  PowerShell:  $env:ROBOFLOW_API_KEY = "<your key>"\n'
            "  bash:        export ROBOFLOW_API_KEY=<your key>"
        )
    return key


def load_model(version: int):
    """Connect to the hosted RF-DETR model.

    Uses the `roboflow` package rather than `inference_sdk` because this is the
    client that produced the detection results reported in the thesis.
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=get_api_key())
    try:
        project = rf.workspace(WORKSPACE).project(PROJECT)
    except Exception:
        project = rf.workspace().project(PROJECT)
    return project.version(version).model


def collect_inputs(path: Path) -> list:
    """Return an ordered list of image paths from a file or a directory."""
    if path.is_file():
        return [str(path)]
    files = []
    for suffix in IMAGE_SUFFIXES:
        files.extend(glob.glob(str(path / f"*{suffix}")))
    if not files:
        sys.exit(f"No images found in {path}")
    return sorted(files)


def predict_polygons(model, image_path: str, confidence: int) -> list:
    """Run the detector and return [(class_name, confidence, Nx2 int array)]."""
    result = model.predict(image_path, confidence=confidence)
    out = []
    for pred in result.json().get("predictions", []):
        raw = pred.get("points", [])
        if len(raw) < 3:
            continue
        pts = np.array([[int(p["x"]), int(p["y"])] for p in raw], dtype=np.int32)
        out.append((pred.get("class", "unknown"), pred.get("confidence", 0.0), pts))
    return out


def reject_oversized(polygons, width, height, max_area_ratio):
    """Drop polygons whose bounding box covers too much of the frame.

    A detection spanning most of the image is the failure mode where the model
    grabs the whole road surface instead of a boundary.
    """
    kept, rejected = [], 0
    for cls, conf, pts in polygons:
        bbox_w = pts[:, 0].max() - pts[:, 0].min()
        bbox_h = pts[:, 1].max() - pts[:, 1].min()
        if (bbox_w * bbox_h) / float(width * height) > max_area_ratio:
            rejected += 1
            continue
        kept.append((cls, conf, pts))
    return kept, rejected


def draw_polygons(img, polygons, alpha=0.35):
    """Draw filled translucent polygons with class and confidence labels."""
    for cls, conf, pts in polygons:
        color = COLORS.get(cls, DEFAULT_COLOR)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        cv2.polylines(img, [pts], True, color, 2)
        top = pts[pts[:, 1].argmin()]
        cv2.putText(img, f"{cls} {conf:.0%}", (int(top[0]), int(top[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img


def write_masks(polygons, width, height, out_dir: Path, stem: str):
    """Write one binary mask per class, which is what src/unproject.py consumes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_class = {}
    for cls, _conf, pts in polygons:
        name = CANONICAL.get(cls, cls)
        mask = by_class.setdefault(name, np.zeros((height, width), np.uint8))
        cv2.fillPoly(mask, [pts], 255)
    for name, mask in by_class.items():
        cv2.imwrite(str(out_dir / f"{stem}_{name}.png"), mask)
    return sorted(by_class)


def run(args):
    files = collect_inputs(args.input)
    print(f"{len(files)} input image(s)")

    first = cv2.imread(files[0])
    if first is None:
        sys.exit(f"Could not read {files[0]}")
    height, width = first.shape[:2]
    print(f"frame size: {width}x{height}")

    model = load_model(args.model_version)
    print(f"model: {PROJECT} v{args.model_version}, confidence >= {args.confidence}")

    writer = None
    if args.output and args.output.suffix.lower() == ".mp4":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (width, height))
        if not writer.isOpened():
            sys.exit(f"Could not open video writer for {args.output}")

    # Temporal smoothing state, only used for sequences written to video.
    mask_ema, conf_ema = {}, {}
    seen_classes, total_rejected = set(), 0

    for idx, image_path in enumerate(files):
        stem = Path(image_path).stem
        img = cv2.imread(image_path)
        if img is None:
            print(f"[{idx + 1}/{len(files)}] {stem}: unreadable, skipped")
            continue

        try:
            polygons = predict_polygons(model, image_path, args.confidence)
        except Exception as exc:
            print(f"[{idx + 1}/{len(files)}] {stem}: ERROR {exc}")
            continue

        polygons, rejected = reject_oversized(polygons, width, height,
                                              args.max_area_ratio)
        total_rejected += rejected
        seen_classes.update(cls for cls, _, _ in polygons)
        print(f"[{idx + 1}/{len(files)}] {stem}: {len(polygons)} kept, "
              f"{rejected} oversized")

        if args.masks_out:
            write_masks(polygons, width, height, args.masks_out, stem)

        if writer is not None:
            img = draw_smoothed(img, polygons, mask_ema, conf_ema,
                                width, height, args.alpha, args.mask_threshold)
            cv2.putText(img, stem, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
        elif args.output is not None:
            cv2.imwrite(str(args.output), draw_polygons(img, polygons))
            print(f"saved {args.output}")

    if writer is not None:
        writer.release()
        print(f"saved {args.output}")

    print(f"classes seen: {sorted(seen_classes)}")
    print(f"oversized rejected: {total_rejected}")


def draw_smoothed(img, polygons, mask_ema, conf_ema, width, height,
                  alpha, threshold):
    """Exponential moving average over per-class masks.

    Raw per-frame detections flicker badly across a sequence. Averaging the
    rasterised masks over time and re-contouring the result is what made the
    output video stable.
    """
    detected = {}
    for cls, conf, pts in polygons:
        mask = detected.setdefault(cls, [np.zeros((height, width), np.float32),
                                         conf])[0]
        cv2.fillPoly(mask, [pts], 1.0)
        detected[cls][1] = max(detected[cls][1], conf)

    for cls in set(mask_ema) | set(detected):
        new_mask, new_conf = detected.get(
            cls, [np.zeros((height, width), np.float32), 0.0])
        prev = mask_ema.get(cls)
        mask_ema[cls] = new_mask if prev is None else alpha * new_mask + (1 - alpha) * prev
        conf_ema[cls] = alpha * new_conf + (1 - alpha) * conf_ema.get(cls, new_conf)

    for cls, smoothed in mask_ema.items():
        binary = (smoothed > threshold).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        color = COLORS.get(cls, DEFAULT_COLOR)
        cv2.polylines(img, contours, True, color, 3)
        biggest = max(contours, key=cv2.contourArea)
        top = biggest[biggest[:, 0, 1].argmin()][0]
        cv2.putText(img, f"{cls} {conf_ema[cls]:.0%}",
                    (int(top[0]), int(top[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR racetrack boundary detection (painted_line, track_edge)")
    parser.add_argument("--input", type=Path, required=True,
                        help="image file or directory of frames")
    parser.add_argument("--output", type=Path,
                        help="output .png (single image) or .mp4 (sequence)")
    parser.add_argument("--masks-out", type=Path,
                        help="directory to write per-class binary masks into")
    parser.add_argument("--model-version", type=int, default=DEFAULT_VERSION,
                        help=f"Roboflow model version (default {DEFAULT_VERSION})")
    parser.add_argument("--confidence", type=int, default=40,
                        help="confidence threshold percent (default 40)")
    parser.add_argument("--max-area-ratio", type=float, default=0.35,
                        help="reject polygons whose bbox exceeds this frame fraction")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="temporal smoothing weight for new frames")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="binarisation threshold on the smoothed mask")
    parser.add_argument("--fps", type=int, default=6, help="output video fps")
    args = parser.parse_args()

    if not args.output and not args.masks_out:
        parser.error("nothing to do: pass --output and/or --masks-out")
    run(args)


if __name__ == "__main__":
    main()
