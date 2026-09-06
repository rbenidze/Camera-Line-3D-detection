# EXPERIMENT: re-test LiDAR projection using the YAML principal point instead of the image centre. RESULT: confirmed cx,cy are not at the centre; using the centre floated points above the road.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
Re-test LiDAR projection using the YAML's ACTUAL principal point (cx, cy),
not the image center. The previous overlay put points too HIGH (floating over
the road), which is the signature of a wrong cy. The fl YAML gives a non-center
principal point; scaled to 1280x720 it shifts points down onto the road.

fl YAML (1506x728): fx=2555.26 fy=2538.43 cx=751.53 cy=469.38
Scale to 1280x720:  sx=1280/1506=0.8500  sy=720/728=0.9890
  fx' = 2555.26*0.8500 = 2172   fy' = 2538.43*0.9890 = 2510  (use fy for vertical)
  cx' = 751.53*0.8500  = 639    cy' = 469.38*0.9890  = 464

We test several fx with the SCALED principal point, plus the scaled-YAML combo
exactly. fy is applied separately from fx (the YAML fx!=fy).

Outputs overlay_P_*.png. The right one puts points ON the road, dense near the
car, smooth red(near)->blue(far).

Run: python test_intrinsics2.py
"""
import os, glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

FRAME_PATH = r"C:\Users\haise\Documents\Camera_line\camera_frames\frame_000020.png"
LIDAR_DIR  = r"C:\Users\haise\Documents\Camera_line\lidar_front"
FRAME_IDX  = 20
W, H = 1280, 720
Z_MIN, Z_MAX = 1.0, 80.0

# (label, fx, fy, cx, cy)
CANDS = [
    ("P_yaml_scaled_full", 2172.0, 2510.0, 639.0, 464.0),   # exact scaled fl YAML
    ("P_fx640_cyYAML",      640.0,  640.0,  639.0, 464.0),   # old fx but correct cy
    ("P_fx369_cyYAML",      369.0,  369.0,  639.0, 464.0),   # CARLA fx, correct cy
    ("P_fx1108_cyYAML",    1108.0, 1108.0,  639.0, 464.0),   # CARLA-at-3840 raw value
]

t_cam = np.array([0.946467741935484, 0.16, 0.344])
r_cam = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
t_lid = np.array([0.951467741935484, 0.0, 0.346])
r_lid = Rotation.from_quat([9.44719270589705e-05, -0.01111577121396888,
                            -9.494350635749763e-07, 0.9999382134434469])

def load_sweep(p):
    a = np.load(p)
    if a.dtype.names: return np.stack([a["x"],a["y"],a["z"]],1).astype(np.float64)
    return np.asarray(a,np.float64)[:,:3]

def to_cam(pts):
    pb = (r_lid.as_matrix() @ pts.T).T + t_lid
    pc = (r_cam.inv().as_matrix() @ (pb - t_cam).T).T
    o = np.empty_like(pc); o[:,0]=-pc[:,1]; o[:,1]=-pc[:,2]; o[:,2]=pc[:,0]
    return o

def main():
    img0 = cv2.imread(FRAME_PATH)
    sweeps = sorted(glob.glob(os.path.join(LIDAR_DIR,"*.npy")),
                    key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))
    sweep = sweeps[min(FRAME_IDX,len(sweeps)-1)]
    pts = to_cam(load_sweep(sweep)); z = pts[:,2]; ok=z>Z_MIN; pts,z=pts[ok],z[ok]
    for label,fx,fy,cx,cy in CANDS:
        img=img0.copy()
        u = fx*pts[:,0]/pts[:,2]+cx
        v = fy*pts[:,1]/pts[:,2]+cy
        m=(u>=0)&(u<W)&(v>=0)&(v<H)&(z<Z_MAX)
        ui,vi,zz=u[m].astype(int),v[m].astype(int),z[m]
        for x,y,d in zip(ui,vi,zz):
            t=np.clip((d-Z_MIN)/(40.0-Z_MIN),0,1)
            c=cv2.applyColorMap(np.uint8([[int(255*(1-t))]]),cv2.COLORMAP_JET)[0][0]
            cv2.circle(img,(x,y),1,(int(c[0]),int(c[1]),int(c[2])),-1)
        out=f"overlay_{label}.png"; cv2.imwrite(out,img)
        print(f"{label}: fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f} -> {m.sum()} pts -> {out}")
    print("\nOpen overlay_P_*.png. Correct one: points ON the road, DENSE near the car,")
    print("smooth near(red)->far(blue). Watch the road surface in the lower half fills in.")

if __name__=="__main__":
    main()
