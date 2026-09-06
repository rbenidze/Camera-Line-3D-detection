"""
Build sparse GT disparity for Yas Marina stereo pairs from LiDAR.

LiDAR is used only during development, to supervise the depth stage. It is not
used at inference.

Reuses the validated transform chain from tools/project_lidar.py:
  lidar -> base_link -> camera (ROS axes) -> optical axes -> project.

WHY THE ESTIMATED INTRINSICS ARE NOT TRUSTED FOR GT SCALE
---------------------------------------------------------
GT disparity must match true pixel correspondences. d = (fx*B)/Z, and only the
PRODUCT fx*B sets that scale. We recover the product from data instead of
assuming it: run pretrained RAFT-Stereo on a few cropped pairs, then fit

      fx*B = median( d_predicted(u,v) * Z_lidar(u,v) )

over thousands of projected LiDAR points. The pretrained model's matching gives
the global scale, LiDAR gives per-point depth.

WHY fxb_recovered IS SEPARATE FROM fx AND baseline IN THE CONFIG
----------------------------------------------------------------
The config carries both, and they are not interchangeable:

  fx, fy, cx, cy, baseline   estimated placement intrinsics. Used to decide
                             WHICH PIXEL a 3D point lands on. Also read by
                             src/unproject.py.
  fxb_recovered              a single number measured from data, used ONLY as
                             the disparity scale here.

They are separate because fx*B is one quantity and fx and baseline are two.
Splitting a recovered product back into a factor pair is underdetermined:
infinitely many (fx, B) give the same product, so writing it back as two fields
would invent a baseline that was never measured. It would also overwrite the
placement intrinsics that src/unproject.py depends on. `recover` therefore
writes one field, `fxb_recovered`, and `generate` reads that field.

Note also that fx_projection (640.0) deliberately differs from fx (1280.0).
The 640.0 value is the one whose overlay was validated. See the config file.

TWO MODES
---------
  recover    Reads predicted-disparity .npy files from --pred-dir (make them
             first by running upstream demo.py on a few cropped pairs), fits
             fx*B, and WRITES it into the calibration JSON as fxb_recovered.
             No copy-paste step.

  generate   Reads fxb_recovered from the calibration JSON (or --fxb to
             override for an experiment). Writes sparse 16-bit GT PNGs
             (KITTI convention, value = disparity*256, 0 = invalid) into
             disp_occ_0/, saves nose-mask and GT-alignment previews, and moves
             every Nth pair into a held-out eval folder.

The holdout split here is a Yas-specific stride over this manifest. It is NOT
the same mechanism as the DrivingStereo holdout built by tools/make_holdout.py
and must not share its manifest format.

Usage:
    python tools/build_lidar_gt.py recover \
        --dst-root ~/RAFT-Stereo/datasets/YasMarina/training \
        --lidar-dir lidar_front --pred-dir ~/RAFT-Stereo/demo_output_yas

    python tools/build_lidar_gt.py generate \
        --dst-root ~/RAFT-Stereo/datasets/YasMarina/training \
        --hold-root ~/RAFT-Stereo/datasets/YasMarina/holdout \
        --lidar-dir lidar_front
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Extrinsics from /tf_static, verbatim from the validated projection.
T_CAM = np.array([0.946467741935484, 0.16, 0.344])
R_CAM = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
T_LID = np.array([0.951467741935484, 0.0, 0.346])
R_LID = Rotation.from_quat([9.44719270589705e-05, -0.01111577121396888,
                            -9.494350635749763e-07, 0.9999382134434469])

DEFAULT_CALIB = Path("config/stereo_yas.json")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def load_calib(path: Path):
    with open(path) as f:
        calib = json.load(f)
    return {
        "fx_projection": calib.get("fx_projection", calib["fx"]),
        "cx": calib["cx"],
        "cy": calib["cy"],
        "width": int(calib.get("image_width", 1280)),
        "height": int(calib.get("image_height", 720)),
        "fxb_recovered": calib.get("fxb_recovered"),
        "nose_rects": [tuple(r) for r in calib.get("nose_rects", [])],
    }


def write_fxb(path: Path, fxb: float):
    """Write fxb_recovered back into the calibration JSON, preserving the rest."""
    with open(path) as f:
        calib = json.load(f)
    previous = calib.get("fxb_recovered")
    calib["fxb_recovered"] = round(float(fxb), 4)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
        f.write("\n")
    if previous is None:
        print(f"[calib] wrote fxb_recovered = {calib['fxb_recovered']} -> {path}")
    else:
        print(f"[calib] fxb_recovered {previous} -> {calib['fxb_recovered']} in {path}")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_manifest(dst_root: Path):
    with open(dst_root / "manifest.json") as f:
        return json.load(f)


def load_sweep(path):
    """Load one LiDAR sweep .npy as (N,3) xyz in the lidar frame."""
    arr = np.load(path)
    if arr.dtype.names:
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    return np.asarray(arr, dtype=np.float64)[:, :3]


def lidar_files_sorted(lidar_dir: Path):
    files = glob.glob(str(lidar_dir / "*.npy"))
    return sorted(files, key=lambda p: int(Path(p).stem))


def build_frame_to_sweep(manifest, sweeps):
    """Map each pair's frame to a LiDAR sweep.

    Preferred: match on the per-pair 'timestamp' written by
    tools/add_timestamps.py, which is robust to the two topics publishing at
    different rates. Index pairing is the fallback and is known to be wrong:
    it produced a 5x spread in recovered fx*B.
    """
    pairs = manifest["pairs"]
    has_ts = bool(pairs) and all("timestamp" in p for p in pairs)

    if has_ts and sweeps:
        sweep_ts = np.array([int(Path(p).stem) for p in sweeps], dtype=np.int64)
        order = np.argsort(sweep_ts)
        sorted_ts = sweep_ts[order]
        mapping, max_dt = {}, 0
        for pair in pairs:
            t = int(pair["timestamp"])
            i = int(np.searchsorted(sorted_ts, t))
            cands = [j for j in (i - 1, i) if 0 <= j < len(sorted_ts)]
            best = min(cands, key=lambda j: abs(int(sorted_ts[j]) - t))
            mapping[pair["kitti_name"]] = sweeps[order[best]]
            max_dt = max(max_dt, abs(int(sorted_ts[best]) - t))
        return mapping, f"timestamp-nearest (max gap {max_dt / 1e6:.1f} ms)"

    mapping = {p["kitti_name"]: sweeps[min(p["frame_index"], len(sweeps) - 1)]
               for p in pairs}
    return mapping, "index-based (no timestamps in manifest; known to be wrong)"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def to_camera_optical(pts_lidar):
    """lidar -> base_link -> camera (ROS axes) -> camera optical axes."""
    pts_base = (R_LID.as_matrix() @ pts_lidar.T).T + T_LID
    pts_ros = (R_CAM.inv().as_matrix() @ (pts_base - T_CAM).T).T
    pts = np.empty_like(pts_ros)
    pts[:, 0] = -pts_ros[:, 1]
    pts[:, 1] = -pts_ros[:, 2]
    pts[:, 2] = pts_ros[:, 0]
    return pts


def project_full(pts_cam, fx, cx, cy, z_min):
    """Project camera-frame points to full-frame pixels."""
    ok = pts_cam[:, 2] > z_min
    p = pts_cam[ok]
    u = fx * p[:, 0] / p[:, 2] + cx
    v = fx * p[:, 1] / p[:, 2] + cy
    return u, v, p[:, 2]


def in_nose_mask(u, v, nose_rects):
    mask = np.zeros_like(u, dtype=bool)
    for (x1, y1, x2, y2) in nose_rects:
        mask |= (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
    return mask


def valid_full_frame(u, v, z, cfg, z_max):
    ok = ((u >= 0) & (u < cfg["width"]) & (v >= 0) & (v < cfg["height"])
          & (z < z_max))
    return ok & ~in_nose_mask(u, v, cfg["nose_rects"])


# --------------------------------------------------------------------------
# recover
# --------------------------------------------------------------------------
def recover(args, cfg, manifest):
    """fx*B = median(d_pred * Z) over LiDAR points that have predicted disparity."""
    crop_top = manifest["crop_top"]
    sweeps = lidar_files_sorted(args.lidar_dir)
    print(f"[lidar] {len(sweeps)} sweeps")
    frame_to_sweep, how = build_frame_to_sweep(manifest, sweeps)
    print(f"[match] frame->sweep matching: {how}")

    preds = sorted(glob.glob(str(args.pred_dir / "*_10.npy")))
    if not preds:
        sys.exit(
            f"No predicted disparities in {args.pred_dir}.\n"
            "Run upstream demo.py on a few cropped pairs first, e.g.\n"
            "  python demo.py --restore_ckpt models/raftstereo-sceneflow.pth "
            "--save_numpy \\\n"
            "    -l datasets/YasMarina/training/image_2/000050_10.png \\\n"
            "    -r datasets/YasMarina/training/image_3/000050_10.png \\\n"
            "    --output_directory demo_output_yas")

    sample = np.abs(np.load(preds[0])).squeeze()
    if sample.shape[1] != cfg["width"]:
        print(f"[warn] predicted disparity width {sample.shape[1]} != "
              f"{cfg['width']}. Pixel placement assumes full width. If you only "
              "cropped the top, this is fine.")

    per_frame = []
    for pred_path in preds:
        kitti_name = Path(pred_path).stem + ".png"
        if kitti_name not in frame_to_sweep:
            print(f"  [skip] {kitti_name} not in manifest")
            continue

        d_pred = np.abs(np.load(pred_path)).squeeze()  # RAFT outputs negative
        crop_h, crop_w = d_pred.shape

        pts = to_camera_optical(load_sweep(frame_to_sweep[kitti_name]))
        u, v, z = project_full(pts, cfg["fx_projection"], cfg["cx"], cfg["cy"],
                               args.z_min)
        ok = valid_full_frame(u, v, z, cfg, args.z_max)
        u, v, z = u[ok], v[ok], z[ok]

        vc = v - crop_top
        ok2 = (vc >= 0) & (vc < crop_h) & (u < crop_w)
        ui, vi, z = u[ok2].astype(int), vc[ok2].astype(int), z[ok2]

        dp = d_pred[vi, ui]
        good = dp > 0.5
        products = dp[good] * z[good]
        if len(products) < args.min_points:
            print(f"  [warn] {kitti_name}: only {len(products)} usable points, skipped")
            continue

        median = float(np.median(products))
        per_frame.append(median)
        print(f"  {kitti_name}: fx*B ~= {median:.2f}   ({len(products)} points)")

    if not per_frame:
        sys.exit("Recovery failed, no usable frames.")

    fxb = float(np.median(per_frame))
    spread = float(np.max(per_frame) - np.min(per_frame))
    print(f"\n[RESULT] fx*B = {fxb:.2f}   (spread across frames: {spread:.2f})")
    if spread > 0.1 * fxb:
        print("[warn] large spread between frames. Check gt_alignment_preview.png "
              "after generate before trusting this value.")

    if args.dry_run:
        print("[dry-run] not writing to the calibration file")
        return
    write_fxb(args.calib, fxb)


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------
def generate(args, cfg, manifest):
    fxb = args.fxb if args.fxb is not None else cfg["fxb_recovered"]
    if fxb is None:
        sys.exit(
            f"fxb_recovered is null in {args.calib}.\n"
            "Run 'build_lidar_gt.py recover' first, or pass --fxb to override.")
    source = "--fxb override" if args.fxb is not None else str(args.calib)
    print(f"[calib] fx*B = {fxb} (from {source})")

    crop_top, crop_bottom = manifest["crop_top"], manifest["crop_bottom"]
    crop_h = crop_bottom - crop_top

    # Idempotency guard: a previous run moves pairs out to the holdout, so a
    # second run would find them missing and silently write partial output.
    disp_dir = args.dst_root / "disp_occ_0"
    missing = [p["kitti_name"] for p in manifest["pairs"]
               if not (args.dst_root / "image_2" / p["kitti_name"]).exists()]
    if missing and not args.force:
        sys.exit(
            f"{len(missing)} of {len(manifest['pairs'])} manifest pairs are "
            f"missing from {args.dst_root / 'image_2'} "
            f"(e.g. {', '.join(missing[:3])}).\n"
            "This usually means generate has already run and moved them to the "
            "holdout. Re-stage with tools/sample_stereo_pairs.py, or pass "
            "--force to process only what is still present.")
    if missing:
        print(f"[force] {len(missing)} pairs missing, processing the remaining "
              f"{len(manifest['pairs']) - len(missing)}")

    sweeps = lidar_files_sorted(args.lidar_dir)
    frame_to_sweep, how = build_frame_to_sweep(manifest, sweeps)
    print(f"[match] frame->sweep matching: {how}")
    disp_dir.mkdir(parents=True, exist_ok=True)

    _write_nose_preview(args, cfg, manifest, crop_top)

    gt_preview_done, written = False, 0
    for pair in manifest["pairs"]:
        if not (args.dst_root / "image_2" / pair["kitti_name"]).exists():
            continue

        pts = to_camera_optical(load_sweep(frame_to_sweep[pair["kitti_name"]]))
        u, v, z = project_full(pts, cfg["fx_projection"], cfg["cx"], cfg["cy"],
                               args.z_min)
        ok = valid_full_frame(u, v, z, cfg, args.z_max)
        u, v, z = u[ok], v[ok], z[ok]

        vc = v - crop_top
        ok2 = (vc >= 0) & (vc < crop_h)
        ui, vi, z = u[ok2].astype(int), vc[ok2].astype(int), z[ok2]

        # z-buffer, nearest point wins each pixel
        zbuf = np.full((crop_h, cfg["width"]), np.inf, dtype=np.float64)
        np.minimum.at(zbuf, (vi, ui), z)

        disp = np.zeros((crop_h, cfg["width"]), dtype=np.float64)
        hit = np.isfinite(zbuf)
        disp[hit] = fxb / zbuf[hit]

        out16 = np.clip(disp * 256.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(disp_dir / pair["kitti_name"]), out16)
        written += 1

        if not gt_preview_done:
            _write_gt_preview(args, pair, disp, hit)
            gt_preview_done = True

    print(f"[gt] wrote {written} sparse disparity maps -> {disp_dir}")

    if args.holdout_every and args.holdout_every > 0:
        _split_holdout(args, manifest)

    print("[check] nose_mask_preview.png: red boxes must cover wheel, nose and antenna")
    print("[check] gt_alignment_preview.png: coloured dots must sit ON road and walls,")
    print("        colours smooth near to far. If dots float off geometry, stop.")


def _write_nose_preview(args, cfg, manifest, crop_top):
    first = manifest["pairs"][0]
    src = Path(manifest["src_left"]) / f"frame_{first['frame_index']:06d}.png"
    preview = cv2.imread(str(src))
    if preview is None:
        print(f"[warn] could not read {src} for the nose-mask preview")
        return
    for (x1, y1, x2, y2) in cfg["nose_rects"]:
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.line(preview, (0, crop_top), (cfg["width"], crop_top), (0, 255, 0), 2)
    cv2.imwrite(str(args.dst_root / "nose_mask_preview.png"), preview)


def _write_gt_preview(args, pair, disp, hit):
    left = cv2.imread(str(args.dst_root / "image_2" / pair["kitti_name"]))
    if left is None or not hit.any():
        return
    ys, xs = np.where(hit)
    dmax = disp[hit].max()
    colors = cv2.applyColorMap(
        (255 * disp[ys, xs] / dmax).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET).reshape(-1, 3)
    for (x, y, c) in zip(xs, ys, colors):
        cv2.circle(left, (int(x), int(y)), 1,
                   (int(c[0]), int(c[1]), int(c[2])), -1)
    cv2.imwrite(str(args.dst_root / "gt_alignment_preview.png"), left)


def _split_holdout(args, manifest):
    """Move every Nth triple out of training. Yas-specific, stride over this
    manifest only. Not the DrivingStereo holdout mechanism."""
    for sub in ("image_2", "image_3", "disp_occ_0"):
        (args.hold_root / sub).mkdir(parents=True, exist_ok=True)
    moved = 0
    for i, pair in enumerate(manifest["pairs"]):
        if i % args.holdout_every != 0:
            continue
        for sub in ("image_2", "image_3", "disp_occ_0"):
            src = args.dst_root / sub / pair["kitti_name"]
            if src.exists():
                shutil.move(str(src), str(args.hold_root / sub / pair["kitti_name"]))
        moved += 1
    print(f"[holdout] moved {moved} pairs -> {args.hold_root} (never train on these)")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build sparse LiDAR-derived GT disparity for Yas stereo pairs.")
    parser.add_argument("mode", choices=("recover", "generate"))
    parser.add_argument("--dst-root", type=Path, required=True,
                        help="training directory holding manifest.json, "
                             "image_2/ and image_3/")
    parser.add_argument("--lidar-dir", type=Path, required=True,
                        help="directory of {timestamp_ns}.npy LiDAR sweeps")
    parser.add_argument("--pred-dir", type=Path,
                        help="recover mode: directory of demo.py *_10.npy "
                             "predicted disparities")
    parser.add_argument("--hold-root", type=Path,
                        help="generate mode: destination for the held-out split")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                        help=f"calibration JSON (default {DEFAULT_CALIB})")
    parser.add_argument("--fxb", type=float,
                        help="override fxb_recovered for an experiment; does "
                             "not write to the calibration file")
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=80.0)
    parser.add_argument("--holdout-every", type=int, default=17,
                        help="generate mode: move every Nth pair to the "
                             "holdout (default 17; 0 disables)")
    parser.add_argument("--min-points", type=int, default=100,
                        help="recover mode: skip a frame with fewer usable "
                             "points (default 100)")
    parser.add_argument("--force", action="store_true",
                        help="generate mode: proceed even if manifest pairs "
                             "are already missing from the training tree")
    parser.add_argument("--dry-run", action="store_true",
                        help="recover mode: print fx*B without writing it")
    args = parser.parse_args()

    if args.mode == "recover" and args.pred_dir is None:
        parser.error("recover mode needs --pred-dir")
    if args.mode == "generate" and args.holdout_every and args.hold_root is None:
        parser.error("generate mode needs --hold-root (or --holdout-every 0)")

    cfg = load_calib(args.calib)
    manifest = load_manifest(args.dst_root)

    if args.mode == "recover":
        recover(args, cfg, manifest)
    else:
        generate(args, cfg, manifest)


if __name__ == "__main__":
    main()
