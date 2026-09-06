# EXPERIMENT: estimate stereo rectification for Yas from URDF poses and an assumed field of view. RESULT: failed. Rows align but metric scale stays wrong; disparity inflated roughly 10x.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
FAILED ATTEMPT. This is not a working calibration stage and nothing in the
pipeline depends on it. It ships as a record of what was tried.

What it does: a full 3-axis (pitch, roll, yaw) corrective search over the Yas
fl/fr pair, evaluated across several frames, minimising the residual vertical
offset between matched features, coarse then fine. Version 3 of five attempts;
the earlier four are in _archive/.

Why it failed: rectifying from estimated parameters cannot recover metric scale
that was never measured. The intrinsics here come from an assumed pinhole field
of view, and the mounting poses come from the URDF, because the recording ships
no calibration files. A pair rectified this way yields disparity inflated by
roughly 10x, and Yas Marina point clouds built from it come out compressed 4 to
5 times in metric scale. Aligning rows is not the same as recovering geometry:
the search can drive the vertical offset down and still leave the horizontal
scale wrong, because nothing in the objective constrains it.

Every metrically valid number in the thesis comes from DrivingStereo, which
ships real calibration. See docs/calibration-limitation.md.

Two further limits worth knowing before reading anything into its output:
  - The objective only measures vertical alignment. A low mean|dy| says rows
    match, not that the stereo geometry is right.
  - --scan-limit caps how far into the recording candidate frames are drawn
    from, so the default estimate is based on the opening frames rather than
    the whole run.

Outputs:
  <out>                     best pair side by side with row guides
  <out-left> / <out-right>  rectified pair, as fed to RAFT-Stereo at the time
  prints the best pitch/roll/yaw and the residual offset

Usage:
    python tools/rectify_yas.py --left-dir camera_frames --right-dir camera_frames_fr \
        --out yas_rectified_v3.png \
        --out-left yas_fl_rect.png --out-right yas_fr_rect.png
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Mounting poses from the URDF, in degrees.
FL_XYZ, FL_RPY = [0.946468, 0.16, 0.344], [-91.30, -0.5, -86.5]
FR_XYZ, FR_RPY = [0.946468, -0.16, 0.344], [-90.0, 0.0, -90.0]


def pose(xyz, rpy):
    return (Rotation.from_euler("xyz", rpy, degrees=True).as_matrix(),
            np.array(xyz, np.float64))


def make_detector():
    try:
        return cv2.SIFT_create(nfeatures=3000), cv2.NORM_L2
    except Exception:
        return cv2.ORB_create(3000), cv2.NORM_HAMMING


def pick_frames(left_dir, right_dir, count, scan_limit):
    """Pick `count` frame pairs present in both directories, spread out.

    Candidates are only collected from the first `scan_limit` common frames, so
    the samples are spread across that window rather than the whole recording.
    """
    pairs = []
    for left in sorted(glob.glob(os.path.join(left_dir, "*.png"))):
        right = os.path.join(right_dir, os.path.basename(left))
        if os.path.exists(right):
            pairs.append((left, right))
        if len(pairs) >= scan_limit:
            break
    if len(pairs) > count:
        idx = np.linspace(0, len(pairs) - 1, count).astype(int)
        pairs = [pairs[i] for i in idx]
    return pairs


def maps_for(pitch, roll, yaw, K, D, size, R_rel, T_rel):
    Rc = Rotation.from_euler("xyz", [pitch, roll, yaw],
                             degrees=True).as_matrix() @ R_rel
    R1, R2, P1, P2, _Q, _, _ = cv2.stereoRectify(
        K, D, K, D, size, Rc, T_rel, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return (cv2.initUndistortRectifyMap(K, D, R1, P1, size, cv2.CV_32FC1),
            cv2.initUndistortRectifyMap(K, D, R2, P2, size, cv2.CV_32FC1))


def dy_for_frame(left_gray, right_gray, detector, norm):
    """Median vertical offset between matched features, or None if too few."""
    k1, d1 = detector.detectAndCompute(left_gray, None)
    k2, d2 = detector.detectAndCompute(right_gray, None)
    if d1 is None or d2 is None:
        return None
    knn = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    _F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return None
    inliers = mask.ravel() == 1
    if inliers.sum() < 10:
        return None
    return float(np.median(p2[inliers, 1] - p1[inliers, 1]))


def main():
    parser = argparse.ArgumentParser(
        description="FAILED ATTEMPT, kept as a record. Estimation-based "
                    "rectification of the Yas fl/fr pair. Rectifying from "
                    "estimated parameters did not recover usable metric scale; "
                    "nothing in the pipeline depends on this. See "
                    "docs/calibration-limitation.md.")
    parser.add_argument("--left-dir", required=True, help="fl frame directory")
    parser.add_argument("--right-dir", required=True, help="fr frame directory")
    parser.add_argument("--out", type=Path, default=Path("yas_rectified_v3.png"),
                        help="side-by-side preview with row guides")
    parser.add_argument("--out-left", type=Path, default=Path("yas_fl_rect.png"),
                        help="rectified left image for RAFT-Stereo")
    parser.add_argument("--out-right", type=Path, default=Path("yas_fr_rect.png"),
                        help="rectified right image for RAFT-Stereo")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=120.0,
                        help="assumed horizontal field of view in degrees "
                             "(default 120, the CARLA pinhole assumption)")
    parser.add_argument("--frames", type=int, default=5,
                        help="frame pairs used for the robust estimate (default 5)")
    parser.add_argument("--scan-limit", type=int, default=40,
                        help="how many common frames to draw candidates from, "
                             "counting from the start of the recording "
                             "(default 40; raise it to sample the whole run)")
    parser.add_argument("--coarse-step", type=float, default=2.0,
                        help="coarse search step in degrees (default 2.0)")
    parser.add_argument("--fine-step", type=float, default=0.5,
                        help="fine search step in degrees (default 0.5)")
    args = parser.parse_args()

    W, H = args.width, args.height
    size = (W, H)

    fx = (W / 2.0) / np.tan(np.radians(args.fov / 2.0))
    K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1]], np.float64)
    D = np.zeros(5)
    print(f"[calib] assumed fov={args.fov} deg -> fx={fx:.1f} (estimated, "
          "not measured)")

    R_fl, t_fl = pose(FL_XYZ, FL_RPY)
    R_fr, t_fr = pose(FR_XYZ, FR_RPY)
    R_rel = R_fr.T @ R_fl
    T_rel = R_fr.T @ (t_fl - t_fr)

    detector, norm = make_detector()
    frames = pick_frames(args.left_dir, args.right_dir, args.frames,
                         args.scan_limit)
    if not frames:
        raise SystemExit(
            f"No frame pairs common to {args.left_dir} and {args.right_dir}")
    print(f"using {len(frames)} frame pairs drawn from the first "
          f"{args.scan_limit} common frames")

    loaded = [(cv2.resize(cv2.imread(l), size), cv2.resize(cv2.imread(r), size))
              for l, r in frames]

    def score(pitch, roll, yaw):
        (m1x, m1y), (m2x, m2y) = maps_for(pitch, roll, yaw, K, D, size,
                                          R_rel, T_rel)
        offsets = []
        for left, right in loaded:
            lr = cv2.remap(left, m1x, m1y, cv2.INTER_LINEAR)
            rr = cv2.remap(right, m2x, m2y, cv2.INTER_LINEAR)
            dy = dy_for_frame(cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY),
                              detector, norm)
            if dy is not None:
                offsets.append(abs(dy))
        return float(np.mean(offsets)) if offsets else 9e9

    print("coarse 3-axis search (this takes a minute)...")
    best = (9e9, 0.0, 0.0, 0.0)
    for pitch in np.arange(-12, 12.1, args.coarse_step):
        for roll in np.arange(-8, 8.1, args.coarse_step):
            for yaw in np.arange(-4, 4.1, args.coarse_step):
                s = score(pitch, roll, yaw)
                if s < best[0]:
                    best = (s, pitch, roll, yaw)
    print(f"coarse best: pitch={best[1]:+.0f} roll={best[2]:+.0f} "
          f"yaw={best[3]:+.0f} -> mean|dy|={best[0]:.1f}px")

    _s0, p0, r0, y0 = best
    for pitch in np.arange(p0 - 2, p0 + 2.01, args.fine_step):
        for roll in np.arange(r0 - 2, r0 + 2.01, args.fine_step):
            for yaw in np.arange(y0 - 2, y0 + 2.01, args.fine_step):
                s = score(pitch, roll, yaw)
                if s < best[0]:
                    best = (s, pitch, roll, yaw)
    s0, p0, r0, y0 = best
    print(f"fine best:   pitch={p0:+.2f} roll={r0:+.2f} yaw={y0:+.2f} "
          f"-> mean|dy|={s0:.1f}px")

    (m1x, m1y), (m2x, m2y) = maps_for(p0, r0, y0, K, D, size, R_rel, T_rel)
    left0, right0 = loaded[0]
    left_rect = cv2.remap(left0, m1x, m1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right0, m2x, m2y, cv2.INTER_LINEAR)

    both = np.hstack([left_rect, right_rect])
    for y in range(0, H, 40):
        cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 0), 1)
    cv2.putText(both, "fl rectified", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(both, "fr rectified", (W + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for path, image in ((args.out, both), (args.out_left, left_rect),
                        (args.out_right, right_rect)):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        print(f"saved {path}")

    print(f"final mean vertical offset over {len(loaded)} frames: {s0:.1f} px "
          "(under about 3 px is well rectified)")


if __name__ == "__main__":
    main()
