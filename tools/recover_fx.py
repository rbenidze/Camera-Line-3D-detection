"""
Recover the true focal length using LiDAR as a geometric reference.

The stereo road bends with depth because fx is wrong. LiDAR shows the road is
actually flat at a constant height. So we search for the fx that, when used to
unproject the disparity, makes the stereo road FLAT and matched to LiDAR across
all depth bands.

Requires the DISPARITY MAP for frame 0 (the same frame cloud_000000.ply came
from). Set DISPARITY_PATH below.

Output: the fx (and implied baseline) to put in stereo.json so unproject_stereo.py
produces undistorted metric clouds directly.
"""

import glob
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# ---- EDIT THIS: disparity map for frame 0 ----
DISPARITY_PATH = "disparity_000000.npy"   # .npy / .pfm / 16-bit .png

LIDAR_DIR = "lidar_front"
CX, CY = 640.0, 360.0       # principal point (image center, from stereo.json)
BASELINE = 0.32             # current estimate; fx and B are coupled, see note at end

# depth bands (meters) to enforce flatness across
BANDS = [(3, 6), (6, 9), (9, 12)]
X_HALFWIDTH = 2.0

# extrinsics (tf_static)
t_cam = np.array([0.946467741935484, 0.16, 0.344])
r_cam = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
t_lid = np.array([0.951467741935484, 0.0, 0.346])
r_lid = Rotation.from_quat([9.44719270589705e-05, -0.01111577121396888,
                            -9.494350635749763e-07, 0.9999382134434469])


def load_disparity(path):
    p = path.lower()
    if p.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if p.endswith(".png"):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) / 256.0
    if p.endswith(".pfm"):
        with open(path, "rb") as f:
            hdr = f.readline().decode().rstrip()
            ch = 3 if hdr == "PF" else 1
            w, h = map(int, f.readline().decode().split())
            sc = float(f.readline().decode().rstrip())
            data = np.fromfile(f, ("<" if sc < 0 else ">") + "f")
            shp = (h, w, ch) if ch == 3 else (h, w)
            return np.flipud(np.reshape(data, shp))
    raise ValueError("unsupported disparity format")


def lidar_to_camera(p):
    pb = (r_lid.as_matrix() @ p.T).T + t_lid
    pc = (r_cam.inv().as_matrix() @ (pb - t_cam).T).T
    out = np.zeros_like(pc)
    out[:, 0] = -pc[:, 1]; out[:, 1] = -pc[:, 2]; out[:, 2] = pc[:, 0]
    return out


def unproject(disp, fx, fy, baseline, min_disp=0.5):
    H, W = disp.shape
    valid = np.isfinite(disp) & (disp > min_disp)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = np.zeros_like(disp, np.float32)
    Z[valid] = (fx * baseline) / disp[valid]
    X = (u - CX) * Z / fx
    Y = (v - CY) * Z / fy
    return np.stack([X, Y, Z], -1)[valid]


def road_y_per_band(pts):
    """Median road height in each band (None if too sparse)."""
    out = []
    for zn, zf in BANDS:
        m = (np.abs(pts[:, 0]) < X_HALFWIDTH) & (pts[:, 2] > zn) & (pts[:, 2] < zf)
        out.append(np.percentile(pts[m, 1], 90) if m.sum() > 50 else None)
    return out


def main():
    disp = load_disparity(DISPARITY_PATH)
    print(f"[disp] {disp.shape}")

    # LiDAR reference: the true flat road height
    lid = lidar_to_camera(np.load(sorted(glob.glob(os.path.join(LIDAR_DIR, "*.npy")))[0])[:, :3])
    lid = lid[lid[:, 2] > 0.5]
    lid_road = [v for v in road_y_per_band(lid) if v is not None]
    true_road_y = float(np.mean(lid_road))
    print(f"[lidar] true road height (flat): {true_road_y:.3f} m  "
          f"(bands: {[round(v,3) for v in lid_road]})")

    # search fx so the stereo road is (a) flat across bands and (b) ~= true height
    print("\n[search] sweeping fx...")
    best = None
    for fx in np.arange(400, 3000, 10.0):
        fy = fx  # square pixels (fx=fy in your stereo.json)
        pts = unproject(disp, fx, fy, BASELINE)
        ys = road_y_per_band(pts)
        if any(v is None for v in ys):
            continue
        flatness = np.std(ys)                       # 0 = perfectly flat road
        offset_err = abs(np.mean(ys) - true_road_y) # match to LiDAR height
        cost = flatness + offset_err
        if best is None or cost < best[0]:
            best = (cost, fx, ys, flatness, offset_err)

    cost, fx, ys, flat, off = best
    print(f"\n[result] best fx = {fx:.1f}")
    print(f"         road Y per band: {[round(v,3) for v in ys]}")
    print(f"         flatness (std): {flat:.4f} m   (lower = flatter)")
    print(f"         height error vs LiDAR: {off:.4f} m")
    print(f"\nPut in stereo.json:  fx = fy = {fx:.1f}   (cx={CX}, cy={CY}, baseline={BASELINE})")
    print("NOTE: fx and baseline are coupled (Z = fx*B/disp). This solves for fx")
    print("at the current baseline. If you later get the TRUE baseline, rescale fx by")
    print("the same ratio. The road FLATNESS result is what matters - it removes the")
    print("depth distortion that a single scale factor could not.")


if __name__ == "__main__":
    main()
