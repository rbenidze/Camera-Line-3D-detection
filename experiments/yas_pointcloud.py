# EXPERIMENT: build a Yas cloud from fr images with fl intrinsics, testing whether the fr calibration is simply wrong. RESULT: produced yas_pointcloud.ply; shape plausible, metric scale still wrong.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
Build a 3D point cloud from the Yas fl/fr stereo pair, TESTING the hypothesis that
the fr calibration file is wrong and fr actually matches fl. So we use fr's IMAGES
(right view) with fl's INTRINSICS for both cameras, plus the URDF baseline 0.32 m.

Pipeline: fl(left)+fr(right) -> RAFT-Stereo -> disparity -> depth -> 3D cloud,
colored with the fl image. Saved as yas_pointcloud.ply (open in Windows 3D Viewer
or MeshLab) -- rotate it and check: does the road lie flat? do walls stand up?
are depths sensible? If yes, fr-images + fl-intrinsics works and Yas is salvageable.

Runs at HALF resolution (640x360) to fit a 4 GB GPU.

Place in C:\\Users\\haise\\Downloads and run:  python yas_pointcloud.py
Requires the RAFT-Stereo repo at C:\\Users\\haise\\RAFT-Stereo (for core modules).
"""
import os, sys
RAFT_DIR = r"C:\Users\haise\RAFT-Stereo"
sys.path.insert(0, RAFT_DIR)
sys.path.insert(0, os.path.join(RAFT_DIR, "core"))
import numpy as np, cv2, torch
from argparse import Namespace
from raft_stereo import RAFTStereo
from utils.utils import InputPadder

# ---- inputs ----
L_PATH = r"C:\Users\haise\Documents\Camera_line\camera_frames\frame_000050.png"   # fl = left
R_PATH = r"C:\Users\haise\Documents\Camera_line\camera_frames_fr\frame_000050.png" # fr = right
CKPT   = r"C:\Users\haise\Downloads\2000_ds_finetune_v2.pth"
OUT    = r"C:\Users\haise\Downloads\yas_pointcloud.ply"

SCALE = 0.5                  # half-res for 4GB GPU
BASELINE = 0.32             # meters, from URDF (fl at +0.16, fr at -0.16)

# fl intrinsics (camera_fl YAML, 1506x728) scaled to the ACTUAL 1280x720 frame,
# then scaled again by SCALE for the half-res run.
fx_full = 2555.26 * (1280/1506)   # ~2171.7
fy_full = 2538.43 * (720/728)     # ~2510.5
cx_full = 751.53  * (1280/1506)   # ~638.8
cy_full = 469.38  * (720/728)     # ~464.2

ARGS = Namespace(hidden_dims=[128]*3, corr_implementation='reg', shared_backbone=False,
    corr_levels=4, corr_radius=4, n_downsample=2, context_norm='batch',
    slow_fast_gru=False, n_gru_layers=3, mixed_precision=True)

def load_model(ck):
    m = torch.nn.DataParallel(RAFTStereo(ARGS))
    sd = torch.load(ck, map_location='cuda')
    m.load_state_dict(sd)
    return m.module.cuda().eval()

def read(p):
    im = cv2.imread(p)
    if im is None: raise FileNotFoundError(p)
    im = cv2.resize(im, (int(1280*SCALE), int(720*SCALE)), interpolation=cv2.INTER_AREA)
    return im

def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2,0,1).float()[None].cuda()

def save_ply(path, pts, cols):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x,y,z),(r,g,b) in zip(pts, cols):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")

def main():
    print("loading model...")
    m = load_model(CKPT)
    Lbgr, Rbgr = read(L_PATH), read(R_PATH)
    H, W = Lbgr.shape[:2]
    fx, fy = fx_full*SCALE, fy_full*SCALE
    cx, cy = cx_full*SCALE, cy_full*SCALE
    print(f"running RAFT-Stereo at {W}x{H} (fl intrinsics for both, baseline {BASELINE} m)...")

    l, r = to_tensor(Lbgr), to_tensor(Rbgr)
    pad = InputPadder(l.shape, divis_by=32); l, r = pad.pad(l, r)
    with torch.no_grad():
        _, up = m(l, r, iters=16, test_mode=True)
    disp = np.abs(pad.unpad(up).cpu().numpy().squeeze())   # pixels
    print(f"disparity range: {disp.min():.1f} .. {disp.max():.1f} px")

    # depth from disparity:  Z = fx * baseline / disp
    valid = disp > 0.5
    Z = np.zeros_like(disp); Z[valid] = fx * BASELINE / disp[valid]
    # keep sensible depths
    valid &= (Z > 0.5) & (Z < 80.0)

    ys, xs = np.where(valid)
    z = Z[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], 1)
    cols = cv2.cvtColor(Lbgr, cv2.COLOR_BGR2RGB)[ys, xs]
    print(f"{len(pts)} points (depths {z.min():.1f}..{z.max():.1f} m)")

    save_ply(OUT, pts, cols)
    print(f"\nsaved {OUT}")
    print("Open it (Windows 3D Viewer or MeshLab) and rotate:")
    print(" - road should be a flat plane sloping away")
    print(" - pit walls should stand up vertically on the sides")
    print(" - depths should grow smoothly with distance")
    print("If the geometry looks right -> fr-images + fl-intrinsics WORKS.")
    print("If it's warped/garbage -> pair not rectified or cameras not matched.")

if __name__ == "__main__":
    main()
