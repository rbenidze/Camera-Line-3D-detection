# EXPERIMENT: project LiDAR onto a Yas frame using the URDF camera_fl mounting rotation. RESULT: found the bug. Every earlier script assumed identity rotation, which is why LiDAR never landed on geometry.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
CORRECTED Yas LiDAR->camera overlay.

THE BUG (found via URDF): every previous script used r_cam = identity, i.e. it
assumed the front-left camera had NO rotation. The URDF says camera_fl is mounted
with rpy = (-91.30, -0.5, -86.5) DEGREES -- a ~90 deg roll + ~90 deg yaw (optical
mounting). Ignoring that rotation is why LiDAR never landed, regardless of
intrinsics or timing (timing was confirmed fine at 28 ms).

URDF joints (both children of base_link):
  lidar_front: xyz=(0.956669, 0, 0.346)   rpy=(0.0, -0.401, 0.0)   [RADIANS]
  camera_fl:   xyz=(0.946468, 0.16, 0.344) rpy=(-91.30, -0.5, -86.5) [DEGREES]

URDF rpy = intrinsic XYZ (roll about X, pitch about Y, yaw about Z), applied as
R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

Transform chain to put a LiDAR point into the camera frame:
  p_base   = R_lidar @ p_lidar + t_lidar
  p_camera = R_cam^T @ (p_base - t_cam)
Then project with the camera intrinsics. The camera frame defined by this rpy is
already optical (z forward), so NO extra manual axis swap.

Frame matched to LiDAR by timestamp (28 ms gap, already verified correct).

Run:  python yas_overlay_fixed.py
"""
import os, glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

FRAME_TS   = 1750757718132307927
FRAME_PATH = r"C:\Users\haise\Documents\Camera_line\camera_frames\frame_000050.png"
LIDAR_DIR  = r"C:\Users\haise\Documents\Camera_line\lidar_front"
W, H = 1280, 720
Z_MIN, Z_MAX = 1.0, 80.0

# front-left intrinsics scaled to 1280x720 (from camera_fl YAML 1506x728)
fx, fy = 2171.7, 2510.5
cx, cy = 638.8, 464.2
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)
DIST = np.array([-0.38385, 0.1615, -0.00085, 0.00053, 0.0], float)

# --- URDF extrinsics, REAL rotations ---
# lidar rpy in radians:
R_lidar = Rotation.from_euler('xyz', [0.0, -0.401, 0.0], degrees=False).as_matrix()
t_lidar = np.array([0.956669, 0.0, 0.346])
# camera rpy in degrees:
R_cam   = Rotation.from_euler('xyz', [-91.30, -0.5, -86.5], degrees=True).as_matrix()
t_cam   = np.array([0.946468, 0.16, 0.344])

def load_sweep(p):
    a = np.load(p)
    if a.dtype.names: return np.stack([a["x"],a["y"],a["z"]],1).astype(float)
    return np.asarray(a,float)[:,:3]

def main():
    img = cv2.imread(FRAME_PATH)
    if img is None:
        print("cannot read", FRAME_PATH); return
    files = glob.glob(os.path.join(LIDAR_DIR,"*.npy"))
    ts = np.array([int(os.path.splitext(os.path.basename(f))[0]) for f in files])
    o = np.argsort(ts); ts, files = ts[o], [files[i] for i in o]
    j = int(np.argmin(np.abs(ts - FRAME_TS)))
    print(f"sweep {os.path.basename(files[j])}  gap {abs(int(ts[j])-FRAME_TS)/1e6:.1f} ms")

    p_lidar = load_sweep(files[j])
    # lidar -> base -> camera
    p_base = (R_lidar @ p_lidar.T).T + t_lidar
    p_cam  = (R_cam.T @ (p_base - t_cam).T).T

    img_ud = cv2.undistort(img, K, DIST)
    z = p_cam[:,2]
    fwd = z > Z_MIN
    pc, z = p_cam[fwd], z[fwd]
    u = fx*pc[:,0]/pc[:,2] + cx
    v = fy*pc[:,1]/pc[:,2] + cy
    m = (u>=0)&(u<W)&(v>=0)&(v<H)&(z<Z_MAX)
    ui,vi,zz = u[m].astype(int), v[m].astype(int), z[m]
    print(f"points on image: {m.sum()} of {len(p_lidar)}")

    for x,y,d in zip(ui,vi,zz):
        t = np.clip((d-Z_MIN)/(40.0-Z_MIN),0,1)
        c = cv2.applyColorMap(np.uint8([[int(255*(1-t))]]), cv2.COLORMAP_JET)[0][0]
        cv2.circle(img_ud,(x,y),1,(int(c[0]),int(c[1]),int(c[2])),-1)
    cv2.imwrite("yas_overlay_fixed.png", img_ud)
    print("saved yas_overlay_fixed.png  -- points should now sit ON road/walls")

if __name__ == "__main__":
    main()
