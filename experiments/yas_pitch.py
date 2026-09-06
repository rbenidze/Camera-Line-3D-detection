# EXPERIMENT: sweep an added downward pitch to centre projected LiDAR vertically. RESULT: found the pitch that brings v back into the 0..720 range.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
xyz R^T has points horizontally in-frame (u spans 0..1280) but vertically jammed
at the TOP (v up to ~76, i.e. ~400px too high). Negated angles overshoot to the
bottom. So we need a small downward pitch correction. Sweep an extra pitch added
to the camera before transforming, find what centers v in [0,720], save overlay.
"""
import os, glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

FRAME_TS=1750757718132307927
FRAME_PATH=r"C:\Users\haise\Documents\Camera_line\camera_frames\frame_000050.png"
LIDAR_DIR=r"C:\Users\haise\Documents\Camera_line\lidar_front"
fx,fy,cx,cy=2171.7,2510.5,638.8,464.2
t_lidar=np.array([0.956669,0.0,0.346]); t_cam=np.array([0.946468,0.16,0.344])
R_lidar=Rotation.from_euler('xyz',[0.0,-0.401,0.0]).as_matrix()

def load(p):
    a=np.load(p)
    if a.dtype.names: return np.stack([a["x"],a["y"],a["z"]],1).astype(float)
    return np.asarray(a,float)[:,:3]

def proj(p_lidar, extra_pitch_deg):
    # base rotation, then apply an extra rotation about the camera's X axis (pitch in image)
    Rc = Rotation.from_euler('xyz',[-91.30,-0.5,-86.5],degrees=True).as_matrix()
    # extra pitch about camera x-axis (vertical tilt of optical axis)
    Rp = Rotation.from_euler('x', extra_pitch_deg, degrees=True).as_matrix()
    pb=(R_lidar@p_lidar.T).T+t_lidar
    pc=(Rp @ (Rc.T @ (pb-t_cam).T)).T
    z=pc[:,2]; fwd=z>1.0; pc,z=pc[fwd],z[fwd]
    u=fx*pc[:,0]/pc[:,2]+cx; v=fy*pc[:,1]/pc[:,2]+cy
    m=(u>=0)&(u<1280)&(v>=0)&(v<720)&(z<80)
    return u,v,z,m

def main():
    files=glob.glob(os.path.join(LIDAR_DIR,"*.npy"))
    ts=np.array([int(os.path.splitext(os.path.basename(f))[0]) for f in files])
    o=np.argsort(ts); ts,files=ts[o],[files[i] for i in o]
    p=load(files[int(np.argmin(np.abs(ts-FRAME_TS)))])
    print("extra_pitch : inbounds  (v median of forward pts)")
    best=(-999,0,None)
    for pitch in range(-40,41,5):
        u,v,z,m=proj(p,pitch)
        # v median of forward, horizontally-in pts
        hu=(u>=0)&(u<1280)
        vm=np.median(v[hu]) if hu.any() else 9e9
        print(f"  {pitch:+3d} deg   : {m.sum():6d}    v_med={vm:8.0f}")
        if m.sum()>best[0]: best=(m.sum(),pitch,(u,v,z,m))
    print(f"\nBEST extra pitch {best[1]:+d} deg, {best[0]} inbounds")
    if best[2] and best[0]>0:
        img=cv2.imread(FRAME_PATH); u,v,z,m=best[2]
        for x,y,d in zip(u[m].astype(int),v[m].astype(int),z[m]):
            t=np.clip((d-1.0)/39.0,0,1)
            c=cv2.applyColorMap(np.uint8([[int(255*(1-t))]]),cv2.COLORMAP_JET)[0][0]
            cv2.circle(img,(x,y),1,(int(c[0]),int(c[1]),int(c[2])),-1)
        cv2.imwrite("yas_pitch_best.png",img)
        print("saved yas_pitch_best.png")

if __name__=="__main__":
    main()
