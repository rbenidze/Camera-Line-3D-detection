"""
Stage 3: unproject stereo disparity into metric 3D points.

    Z = fx * B / d
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

Consolidates:
  unproject_stereo.py   dense cloud over every valid pixel
  extract_border_3d.py  the same math gated to RF-DETR boundary polygons

Difference from extract_border_3d.py: this does not call Roboflow. It reads the
per-class masks written by `src/detect.py --masks-out`, so the detection and
geometry stages stay independent and no API key is needed here.

NOTE ON SCALE: with config/stereo_yas.json the intrinsics and baseline are
estimates, so the metric scale carries the known 4-5x error. Shape and relative
geometry are correct, absolute distances are not. Metrically valid output
requires config/stereo_drivingstereo.json, which ships real calibration.

Usage:
    # dense cloud
    python src/unproject.py --disparity d.npy --left frame.png \
        --calib config/stereo_drivingstereo.json --output cloud.ply

    # boundary points only, coloured by class
    python src/unproject.py --disparity d.npy --left frame.png \
        --calib config/stereo_drivingstereo.json --output borders.ply \
        --masks masks/ --frame-stem frame_000050
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# Class colours in the output PLY (RGB, 0..1).
CLASS_COLORS = {
    "painted_line": (1.0, 1.0, 0.0),
    "track_edge": (1.0, 0.5, 0.0),
}
FALLBACK_COLOR = (0.0, 1.0, 0.0)


def load_disparity(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".pfm":
        return _read_pfm(path)
    if suffix == ".png":
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Could not read {path}")
        return raw.astype(np.float32) / 256.0
    raise ValueError(f"Unsupported disparity format: {suffix}")


def _read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            channels = 3
        elif header == "Pf":
            channels = 1
        else:
            raise ValueError("Not a PFM file")
        width, height = map(int, f.readline().decode("utf-8").rstrip().split())
        scale = float(f.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, channels) if channels == 3 else (height, width)
        return np.flipud(np.reshape(data, shape))


def unproject(disparity, fx, fy, cx, cy, baseline, min_disp=0.5):
    """Return (points Nx3, us N, vs N) for every pixel with usable disparity.

    The pixel indices come back alongside the points so callers can match a 3D
    point to the mask that covered its source pixel.
    """
    height, width = disparity.shape
    valid = np.isfinite(disparity) & (disparity > min_disp)
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    Z = np.zeros_like(disparity, dtype=np.float32)
    Z[valid] = (fx * baseline) / disparity[valid]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1)[valid].astype(np.float32)
    return points, u[valid], v[valid], valid


def load_calib(path: Path):
    with open(path) as f:
        calib = json.load(f)
    fx, fy = calib["fx"], calib["fy"]
    cx, cy = calib["cx"], calib["cy"]
    baseline = calib["baseline"]
    print(f"[calib] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} "
          f"B={baseline:.4f} m")
    return fx, fy, cx, cy, baseline


def load_masks(masks_dir: Path, stem: str, shape):
    """Load `<stem>_<class>.png` masks written by src/detect.py."""
    height, width = shape
    masks = {}
    for path in sorted(masks_dir.glob(f"{stem}_*.png")):
        cls = path.stem[len(stem) + 1:]
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask {path.name} is {mask.shape}, expected {(height, width)}")
        masks[cls] = (mask > 127).astype(np.uint8)
    return masks


def save_ply(path: Path, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"[save]  -> {path}  ({len(points):,} pts)")


def main():
    parser = argparse.ArgumentParser(
        description="Unproject stereo disparity to a metric 3D point cloud.")
    parser.add_argument("--disparity", type=Path, required=True,
                        help="disparity map (.npy / .pfm / 16-bit .png)")
    parser.add_argument("--left", type=Path, required=True,
                        help="rectified left RGB image, same resolution")
    parser.add_argument("--calib", type=Path, required=True,
                        help="calibration JSON (fx, fy, cx, cy, baseline)")
    parser.add_argument("--output", type=Path, required=True,
                        help="output .ply path")
    parser.add_argument("--masks", type=Path,
                        help="directory of per-class masks from src/detect.py; "
                             "restricts output to boundary pixels")
    parser.add_argument("--frame-stem",
                        help="frame stem used to find masks (defaults to the "
                             "left image stem)")
    parser.add_argument("--save-npy", action="store_true",
                        help="also write left/right border .npy splits")
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--min-disp", type=float, default=0.5)
    args = parser.parse_args()

    fx, fy, cx, cy, baseline = load_calib(args.calib)

    disp = load_disparity(args.disparity)
    finite = disp[np.isfinite(disp)]
    print(f"[disp]  shape={disp.shape}, "
          f"range=[{finite.min():.2f}, {finite.max():.2f}] px")

    left = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    if left is None:
        raise IOError(f"Could not read {args.left}")
    if left.shape[:2] != disp.shape:
        raise ValueError(
            f"Image shape {left.shape[:2]} != disparity {disp.shape}. "
            f"The left image must be rectified at the same resolution.")

    points, us, vs, valid = unproject(disp, fx, fy, cx, cy, baseline,
                                      min_disp=args.min_disp)
    print(f"[unproj] raw points: {len(points):,}")

    depth_ok = (points[:, 2] >= args.min_depth) & (points[:, 2] <= args.max_depth)
    points, us, vs = points[depth_ok], us[depth_ok], vs[depth_ok]
    print(f"[filter] within [{args.min_depth}, {args.max_depth}] m: "
          f"{len(points):,}")

    if args.masks is None:
        # Dense cloud, coloured from the left image.
        rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        colors = rgb[vs, us].astype(np.float32) / 255.0
        save_ply(args.output, points, colors)
        return

    stem = args.frame_stem or args.left.stem
    masks = load_masks(args.masks, stem, disp.shape)
    if not masks:
        print(f"[warn] no masks matching '{stem}_*.png' in {args.masks}; "
              f"nothing to unproject.")
        return
    print(f"[masks] classes: {sorted(masks)}")

    all_pts, all_cols = [], []
    for cls, mask in sorted(masks.items()):
        inside = mask[vs, us] == 1
        cls_pts = points[inside]
        color = np.array(CLASS_COLORS.get(cls, FALLBACK_COLOR), np.float32)
        all_pts.append(cls_pts)
        all_cols.append(np.tile(color, (len(cls_pts), 1)))
        print(f"[border] {cls}: {len(cls_pts):,} 3D points")

    border_pts = np.concatenate(all_pts, axis=0)
    border_cols = np.concatenate(all_cols, axis=0)
    save_ply(args.output, border_pts, border_cols)

    if args.save_npy:
        # Split left/right by X sign: negative X is left of the camera centre.
        out_dir = args.output.parent
        left_pts = border_pts[border_pts[:, 0] < 0]
        right_pts = border_pts[border_pts[:, 0] >= 0]
        np.save(out_dir / "left_border_3d.npy", left_pts)
        np.save(out_dir / "right_border_3d.npy", right_pts)
        print(f"[save]  -> left_border_3d.npy  ({len(left_pts):,} pts)")
        print(f"[save]  -> right_border_3d.npy ({len(right_pts):,} pts)")


if __name__ == "__main__":
    main()
