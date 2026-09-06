# EXPERIMENT: sweep rpy compose-order variants after the corrected rotation returned zero on-image points. RESULT: identified which convention actually puts points on the image.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
The corrected rotation gave 0 points -> the rotation is right in magnitude but
likely inverted/transposed or the rpy compose order differs. Try the plausible
variants and report how many points land FORWARD (z>0) and ON the image for each.
The correct convention will jump to tens of thousands of on-image points.

URDF: camera_fl rpy=(-91.30,-0.5,-86.5) deg, t=(0.946468,0.16,0.344)
      lidar_front rpy=(0,-0.401,0) rad,       t=(0.956669,0,0.346)
"""
import os, glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

FRAME_TS   = 1750757718132307927
FRAME_PATH = r"C:\Users\haise\Documents\Camera_line\camera_frames\frame_000050.png"
LIDAR_DIR  = r"C:\Users\haise\Documents\Camera_line\lidar_front"
W, H = 1280, 720
fx, fy, cx, cy = 2171.7, 2510.5, 638.8, 464.2

t_lidar = np.array([0.956669, 0.0, 0.346])
t_cam   = np.array([0.946468, 0.16, 0.344])
R_lidar = Rotation.from_euler('xyz', [0.0, -0.401, 0.0], degrees=False).as_matrix()
# camera rotation, several ways to interpret/compose
cam_rpy_deg = [-91.30, -0.5, -86.5]
R_cam_xyz = Rotation.from_euler('xyz', cam_rpy_deg, degrees=True).as_matrix()
R_cam_ZYX = Rotation.from_euler('ZYX', cam_rpy_deg[::-1], degrees=True).as_matrix()

def load_sweep(p):
    a=np.load(p)
    if a.dtype.names: return np.stack([a["x"],a["y"],a["z"]],1).astype(float)
    return np.asarray(a,float)[:,:3]

def count(p_lidar, R_cam, use_T, name):
    p_base = (R_lidar @ p_lidar.T).T + t_lidar
    Rc = R_cam.T if use_T else R_cam
    p_cam = (Rc @ (p_base - t_cam).T).T
    z = p_cam[:,2]; fwd = z>1.0
    pc, zz = p_cam[fwd], z[fwd]
    if len(pc)==0:
        print(f"{name:28s}: 0 forward"); return name, 0, None
    u = fx*pc[:,0]/pc[:,2]+cx; v = fy*pc[:,1]/pc[:,2]+cy
    m=(u>=0)&(u<W)&(v>=0)&(v<H)&(zz<80)
    print(f"{name:28s}: {fwd.sum():6d} forward, {m.sum():6d} on-image")
    return name, int(m.sum()), (u[m],v[m],zz[m])

def main():
    files=glob.glob(os.path.join(LIDAR_DIR,"*.npy"))
    ts=np.array([int(os.path.splitext(os.path.basename(f))[0]) for f in files])
    o=np.argsort(ts); ts,files=ts[o],[files[i] for i in o]
    j=int(np.argmin(np.abs(ts-FRAME_TS)))
    p=load_sweep(files[j])
    print(f"sweep {os.path.basename(files[j])}, {len(p)} pts\n")
    results=[]
    results.append(count(p, R_cam_xyz, True,  "xyz, R^T"))
    results.append(count(p, R_cam_xyz, False, "xyz, R"))
    results.append(count(p, R_cam_ZYX, True,  "ZYX, R^T"))
    results.append(count(p, R_cam_ZYX, False, "ZYX, R"))
    # pick best and save overlay
    best=max(results, key=lambda r:r[1])
    print(f"\nBEST: {best[0]} with {best[1]} on-image points")
    if best[2] is not None and best[1]>0:
        img=cv2.imread(FRAME_PATH)
        u,v,zz=best[2]
        for x,y,d in zip(u.astype(int),v.astype(int),zz):
            t=np.clip((d-1.0)/39.0,0,1)
            c=cv2.applyColorMap(np.uint8([[int(255*(1-t))]]),cv2.COLORMAP_JET)[0][0]
            cv2.circle(img,(x,y),1,(int(c[0]),int(c[1]),int(c[2])),-1)
        cv2.imwrite("yas_overlay_best.png",img)
        print("saved yas_overlay_best.png")

if __name__=="__main__":
    main()
