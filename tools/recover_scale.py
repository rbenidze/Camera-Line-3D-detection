"""
Scale recovery, robust version.

This scene has tall buildings flanking a narrow lane, which defeats plane-fitting
on the full cloud (RANSAC keeps grabbing walls). Fix: restrict BOTH clouds to a
narrow front-center corridor where only the road exists, then fit the ground there.

  LiDAR ground height = TRUE metric reference
  stereo ground height = wrong-scale measurement
  scale = lidar_ground / stereo_ground
"""

import glob
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

STEREO_PLY = "cloud_000000.ply"
LIDAR_DIR = "lidar_front"
OUTPUT_PLY = "cloud_000000_scaled.ply"

# extrinsics (tf_static)
t_cam = np.array([0.946467741935484, 0.16, 0.344])
r_cam = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
t_lid = np.array([0.951467741935484, 0.0, 0.346])
r_lid = Rotation.from_quat([9.44719270589705e-05, -0.01111577121396888,
                            -9.494350635749763e-07, 0.9999382134434469])


def lidar_to_camera(p):
    pb = (r_lid.as_matrix() @ p.T).T + t_lid
    pc = (r_cam.inv().as_matrix() @ (pb - t_cam).T).T
    out = np.zeros_like(pc)
    out[:, 0] = -pc[:, 1]
    out[:, 1] = -pc[:, 2]
    out[:, 2] = pc[:, 0]
    return out


def ground_height_in_corridor(pts, x_halfwidth, z_near, z_far, label):
    """Keep only a narrow front-center corridor (|x|<halfwidth, near<z<far),
    then take the ground as the lower envelope: fit a horizontal plane to the
    lowest band of Y. Camera convention: Y is DOWN, so road = largest Y."""
    m = (np.abs(pts[:, 0]) < x_halfwidth) & (pts[:, 2] > z_near) & (pts[:, 2] < z_far)
    corr = pts[m]
    if len(corr) < 50:
        raise RuntimeError(f"[{label}] corridor too sparse ({len(corr)} pts) "
                           f"- widen the corridor.")
    # ground = high-Y band (bottom of image). Use a robust percentile so a few
    # stray low points don't skew it.
    y = corr[:, 1]
    ground_y = np.percentile(y, 90)        # near the floor, robust to outliers
    print(f"[{label}] corridor pts={len(corr):,}  ground Y={ground_y:.4f}")
    return ground_y


def main():
    # --- stereo ---
    pcd = o3d.io.read_point_cloud(STEREO_PLY)
    stereo = np.asarray(pcd.points)
    print(f"[stereo] {len(stereo):,} pts, "
          f"X[{stereo[:,0].min():.2f},{stereo[:,0].max():.2f}] "
          f"Z[{stereo[:,2].min():.2f},{stereo[:,2].max():.2f}]")
    # corridor sized to the stereo cloud's (compressed) scale
    s_ground = ground_height_in_corridor(
        stereo, x_halfwidth=0.5, z_near=1.5, z_far=4.0, label="stereo")

    # --- lidar (frame 0) ---
    files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.npy")))
    lid = lidar_to_camera(np.load(files[0])[:, :3])
    lid = lid[lid[:, 2] > 0.5]
    print(f"\n[lidar]  {os.path.basename(files[0])}: {len(lid):,} pts (front)")
    # corridor sized to real meters
    l_ground = ground_height_in_corridor(
        lid, x_halfwidth=2.0, z_near=5.0, z_far=20.0, label="lidar")

    # --- scale ---
    # ground Y is the camera-to-road distance (camera origin Y=0, road at +Y)
    scale = l_ground / s_ground
    print(f"\n[scale]  factor = {l_ground:.4f} / {s_ground:.4f} = {scale:.3f}x")

    scaled = stereo * scale
    pcd.points = o3d.utility.Vector3dVector(scaled)
    o3d.io.write_point_cloud(OUTPUT_PLY, pcd)
    print(f"[save]   -> {OUTPUT_PLY}")
    print(f"[check]  new Z depth: {scaled[:,2].min():.2f} -> {scaled[:,2].max():.2f} m")


if __name__ == "__main__":
    main()
