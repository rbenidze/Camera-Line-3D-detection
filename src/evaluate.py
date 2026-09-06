"""
Stage 4: metric 3D reconstruction accuracy on the DrivingStereo holdout.

This is the code that produced the headline numbers in the thesis:

    median 3D error (all pixels):            21.9 cm
    median 3D error (high-gradient pixels):  39.5 cm
    median depth-only error (Z):             21.2 cm

Originally extracted from raft_finetune_colab_v4.ipynb cell 38, which was the
only place it existed. The error arithmetic is unchanged so the figures
reproduce. What changed is how the frame set is chosen.

REPRODUCTION WARNING
--------------------
The reported figures come from the notebook's own split, which re-derived a
stride over whatever was on disk at the time. This script instead reads the
recorded manifest written by tools/make_holdout.py. The two will not
necessarily select the same frames, so a fresh run may differ from the numbers
above. The manifest path is the reproducible one; the notebook figures are the
historical record. Do not overwrite one with the other.

THINGS TO KNOW ABOUT THESE NUMBERS
----------------------------------
1. Aggregation is a mean of per-frame medians. Per frame the code takes
   np.median over valid pixels, then averages those per-frame medians across
   frames with np.mean. It is reported as "median 3D error". That is what
   produced 21.9 cm, so it is kept exactly as is. Do not swap the np.mean for
   an np.median: it would change a locked number.

2. "High-gradient pixels" means exactly that: a Sobel gradient magnitude above
   its 90th percentile. It is NOT the RF-DETR painted_line or track_edge mask.
   The thesis calls this metric "boundary/edge pixels", which overstates what
   is measured. It is a proxy for thin-structure error, which is the point it
   is used to make, but it is measured on image gradient, not on detected
   track boundaries.

Ground truth is the DrivingStereo GT disparity map unprojected with the same
intrinsics, so this measures disparity error propagated into 3D, not detector
error.

Usage:
    python src/evaluate.py --holdout-root datasets/DrivingStereo_holdout \
        --manifest config/holdout_manifest.json \
        --checkpoint checkpoints/2000_ds_finetune_v2.pth \
        --calib config/stereo_drivingstereo.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disparity import add_raft_stereo_to_path, load_model, infer_pair  # noqa: E402

DEFAULT_MANIFEST = Path("config/holdout_manifest.json")


def load_holdout(manifest_path: Path, holdout_root: Path):
    """Resolve every manifest frame to concrete paths, or fail.

    Hard-fails on the first missing file rather than skipping it. A silently
    shrinking evaluation set is how a number stops meaning what it says, and
    falling back to the full dataset would evaluate on frames the model may
    have trained on.
    """
    if not manifest_path.exists():
        sys.exit(
            f"No holdout manifest at {manifest_path}.\n"
            "Create the split first:\n"
            "  python tools/make_holdout.py --root <dataset> "
            "--holdout <holdout> --every 400")

    with open(manifest_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    if not frames:
        sys.exit(f"{manifest_path} lists no frames.")

    resolved, missing = [], []
    for frame in frames:
        left = holdout_root / frame["left"]
        right = holdout_root / frame["right"]
        disp = holdout_root / frame["disparity"]
        absent = [str(p) for p in (left, right, disp) if not p.exists()]
        if absent:
            missing.append((f"{frame['sequence']}/{frame['stem']}", absent))
            continue
        resolved.append((left, right, disp))

    if missing:
        print(f"[FATAL] {len(missing)} of {len(frames)} manifest frames are "
              f"missing from {holdout_root}:", file=sys.stderr)
        for name, absent in missing[:5]:
            print(f"  {name}: {absent[0]}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)
        sys.exit(
            "Refusing to evaluate on a partial holdout. The manifest and the "
            "holdout tree disagree; re-run tools/make_holdout.py or point "
            "--holdout-root at the right tree.")

    print(f"[holdout] {len(resolved)} frames, every {manifest.get('every', '?')} "
          f"of {manifest.get('total_frames_scanned', '?')} scanned")
    return resolved, manifest


def disp_to_3d(disp, fx, fy, cx, cy, baseline):
    """Unproject a disparity map. Mirrors src/unproject.py.

    Y uses fy. The notebook divided Y by fx, which is only correct when the
    pixels are square. For DrivingStereo fx == fy == 1003.556 so this changes
    nothing numerically there, but it is wrong in general and would bite on any
    sensor with non-square pixels.
    """
    height, width = disp.shape
    valid = disp > 0.5
    Z = np.zeros_like(disp)
    Z[valid] = fx * baseline / disp[valid]
    ys, xs = np.mgrid[0:height, 0:width]
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return X, Y, Z, valid


def main():
    parser = argparse.ArgumentParser(
        description="Metric 3D reconstruction accuracy on the DrivingStereo holdout.")
    parser.add_argument("--holdout-root", type=Path, required=True,
                        help="holdout root written by tools/make_holdout.py")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"holdout manifest (default {DEFAULT_MANIFEST})")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="RAFT-Stereo weights to evaluate")
    parser.add_argument("--calib", type=Path, required=True,
                        help="calibration JSON, use config/stereo_drivingstereo.json")
    parser.add_argument("--raft-stereo", type=Path,
                        default=Path(os.environ.get("RAFT_STEREO_PATH",
                                                    Path.home() / "RAFT-Stereo")),
                        help="path to the upstream RAFT-Stereo clone")
    parser.add_argument("--iters", type=int, default=16,
                        help="refinement iterations (default 16, as used for "
                             "the reported figures)")
    parser.add_argument("--max-depth", type=float, default=80.0,
                        help="ignore GT points beyond this depth")
    parser.add_argument("--gradient-percentile", type=float, default=90.0,
                        help="Sobel gradient percentile defining a "
                             "high-gradient pixel (default 90)")
    args = parser.parse_args()

    with open(args.calib) as f:
        calib = json.load(f)
    fx, fy = calib["fx"], calib["fy"]
    cx, cy = calib["cx"], calib["cy"]
    baseline = calib["baseline"]
    print(f"fx={fx}, fy={fy}, baseline={baseline} m, "
          f"depth = {fx * baseline:.1f}/disparity\n")

    frames, manifest = load_holdout(args.manifest, args.holdout_root)

    add_raft_stereo_to_path(args.raft_stereo)
    model = load_model(args.checkpoint)

    all_err, gradient_err, depth_err, all_depths = [], [], [], []

    for left, right, disp_path in frames:
        gt_disp = cv2.imread(str(disp_path),
                             cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        pred = infer_pair(model, left, right, args.iters)
        if pred.shape != gt_disp.shape:
            pred = cv2.resize(pred, (gt_disp.shape[1], gt_disp.shape[0]))

        Xp, Yp, Zp, vp = disp_to_3d(pred, fx, fy, cx, cy, baseline)
        Xg, Yg, Zg, vg = disp_to_3d(gt_disp, fx, fy, cx, cy, baseline)
        valid = vp & vg & (Zg > 0.5) & (Zg < args.max_depth)
        if not valid.any():
            sys.exit(f"No valid pixels in {disp_path}; check the calibration.")

        d3 = np.sqrt((Xp - Xg) ** 2 + (Yp - Yg) ** 2 + (Zp - Zg) ** 2)
        all_err.append(np.median(d3[valid]))
        depth_err.append(np.median(np.abs(Zp[valid] - Zg[valid])))
        all_depths.append(np.median(Zg[valid]))

        # High-gradient pixels: photometric gradient, not detected boundaries.
        gray = cv2.cvtColor(
            cv2.resize(cv2.imread(str(left)),
                       (gt_disp.shape[1], gt_disp.shape[0])), cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        grad = np.sqrt(gx * gx + gy * gy)
        steep = valid & (grad > np.percentile(grad, args.gradient_percentile))
        if steep.sum() > 0:
            gradient_err.append(np.median(d3[steep]))

    # np.mean over per-frame medians. Kept exactly as in the notebook: this is
    # the arithmetic that produced the reported figures.
    print("=== 3D coordinate accuracy (camera vs ground truth) ===")
    print(f"median 3D error (all pixels):            {np.mean(all_err) * 100:.1f} cm")
    print(f"median 3D error (high-gradient pixels):  {np.mean(gradient_err) * 100:.1f} cm")
    print(f"median depth-only error (Z):             {np.mean(depth_err) * 100:.1f} cm")
    print(f"median GT holdout depth:                 {np.median(all_depths):.1f} m "
          f"(range {np.min(all_depths):.1f}-{np.max(all_depths):.1f} m)")
    print(f"\n(mean over {len(frames)} per-frame medians; 3D = Euclidean XYZ "
          f"distance to GT-disparity 3D points)")
    print(f"(holdout: {args.manifest}, created {manifest.get('created', 'unknown')})")
    print("(reported thesis figures came from the notebook's own split; "
          "see the reproduction warning in this file)")


if __name__ == "__main__":
    main()
