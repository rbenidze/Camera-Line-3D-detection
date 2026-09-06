"""
Project a LiDAR sweep onto the matching camera frame and save a depth-coloured
overlay. This is the validated LiDAR to camera transform: if the points land on
real geometry (road on road, walls on walls) the projection is correct.

LiDAR is used only during development to build ground truth. It is not used at
inference.

Extrinsics default to the values read from /tf_static for this recording. The
camera rotation is identity here and the ROS to camera axis conversion is done
explicitly, which is the combination that was validated by overlay on this
recording. See docs/calibration-limitation.md before reusing these numbers on a
different recording.

Usage:
    python tools/project_lidar.py --mcap run.mcap --out lidar_on_camera.png
    python tools/project_lidar.py --mcap run.mcap --calib config/stereo_yas.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

# Estimated intrinsics for the 1280x720 frames. These are not measured; see
# docs/calibration-limitation.md.
DEFAULT_FX = 640.0
DEFAULT_FY = 640.0
DEFAULT_CX = 640.0
DEFAULT_CY = 360.0

# Extrinsics from /tf_static for this recording.
T_CAM = np.array([0.946467741935484, 0.16, 0.344])
Q_CAM = [0.0, 0.0, 0.0, 1.0]
T_LID = np.array([0.951467741935484, 0.0, 0.346])
Q_LID = [9.44719270589705e-05, -0.01111577121396888,
         -9.494350635749763e-07, 0.9999382134434469]

POINT_DTYPE = np.dtype([
    ("x", np.float32), ("y", np.float32), ("z", np.float32),
    ("timestamp", np.float32), ("intensity", np.float32),
])


def read_first_pair(mcap_path, camera_topic, lidar_topic):
    """Return (camera image BGR, lidar points Nx3) from the first of each topic."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    camera_image, lidar_points = None, None

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None:
            raise RuntimeError("No MCAP summary found.")
        schema_map = summary.schemas

        for _schema, channel, message in reader.iter_messages():
            if channel.topic == camera_topic and camera_image is None:
                schema = schema_map[channel.schema_id]
                msg = typestore.deserialize_cdr(message.data, schema.name)
                camera_image = cv2.imdecode(
                    np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)

            elif channel.topic == lidar_topic and lidar_points is None:
                schema = schema_map[channel.schema_id]
                msg = typestore.deserialize_cdr(message.data, schema.name)
                arr = np.frombuffer(msg.data, dtype=POINT_DTYPE,
                                    count=msg.width * msg.height)
                lidar_points = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)

            if camera_image is not None and lidar_points is not None:
                break

    return camera_image, lidar_points


def to_camera_frame(points, r_cam, t_cam, r_lid, t_lid):
    """LiDAR frame -> base_link -> camera frame -> camera optical convention."""
    points_base = (r_lid.as_matrix() @ points.T).T + t_lid
    points_cam = (r_cam.inv().as_matrix() @ (points_base - t_cam).T).T

    # ROS (x forward, y left, z up) -> camera (x right, y down, z forward)
    out = np.zeros_like(points_cam)
    out[:, 0] = -points_cam[:, 1]
    out[:, 1] = -points_cam[:, 2]
    out[:, 2] = points_cam[:, 0]
    return out


def load_intrinsics(calib_path):
    if calib_path is None:
        return DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY
    with open(calib_path) as f:
        calib = json.load(f)
    return calib["fx"], calib["fy"], calib["cx"], calib["cy"]


def main():
    parser = argparse.ArgumentParser(
        description="Project a LiDAR sweep onto the matching camera frame.")
    parser.add_argument("--mcap", type=Path, required=True,
                        help="path to the .mcap recording")
    parser.add_argument("--camera-topic", default="/sensor/camera_fl/compressed_image")
    parser.add_argument("--lidar-topic", default="/sensor/lidar_front/points")
    parser.add_argument("--out", type=Path, default=Path("lidar_on_camera.png"),
                        help="output overlay image (default lidar_on_camera.png)")
    parser.add_argument("--calib", type=Path,
                        help="calibration JSON; defaults to the estimated "
                             "intrinsics for the 1280x720 frames")
    parser.add_argument("--min-depth", type=float, default=1.0,
                        help="drop points closer than this, in metres (default 1.0)")
    parser.add_argument("--radius", type=int, default=2,
                        help="drawn point radius in pixels (default 2)")
    args = parser.parse_args()

    fx, fy, cx, cy = load_intrinsics(args.calib)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if args.calib is None:
        print("[calib] using estimated intrinsics "
              f"(fx={fx}, fy={fy}, cx={cx}, cy={cy}); these are not measured")

    camera_image, lidar_points = read_first_pair(
        args.mcap, args.camera_topic, args.lidar_topic)
    if camera_image is None or lidar_points is None:
        sys.exit("Could not read both a camera frame and a LiDAR sweep. "
                 "Check --camera-topic and --lidar-topic against "
                 "tools/list_topics.py.")

    height, width = camera_image.shape[:2]

    points_camera = to_camera_frame(
        lidar_points,
        Rotation.from_quat(Q_CAM), T_CAM,
        Rotation.from_quat(Q_LID), T_LID)
    points_camera = points_camera[points_camera[:, 2] > args.min_depth]
    print(f"Points in front of camera: {len(points_camera)}")

    points_2d = (K @ points_camera.T).T
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]

    u = points_2d[:, 0].astype(int)
    v = points_2d[:, 1].astype(int)
    depth = points_camera[:, 2]

    # Image bounds come from the frame itself rather than assuming 1280x720.
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, depth = u[valid], v[valid], depth[valid]
    print(f"Points projected onto image: {len(u)}")

    if len(u) == 0:
        print("No points landed on the image. Ranges for debugging:")
        print(f"  camera z: {points_camera[:, 2].min():.2f} to "
              f"{points_camera[:, 2].max():.2f}")
        print(f"  u: {points_2d[:, 0].min():.2f} to {points_2d[:, 0].max():.2f} "
              f"(image width {width})")
        print(f"  v: {points_2d[:, 1].min():.2f} to {points_2d[:, 1].max():.2f} "
              f"(image height {height})")
        return

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    colors = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET).reshape(-1, 3)
    for i in range(len(u)):
        color = colors[i]
        cv2.circle(camera_image, (u[i], v[i]), args.radius,
                   (int(color[0]), int(color[1]), int(color[2])), -1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), camera_image)
    print(f"Saved {args.out} with {len(u)} projected points")


if __name__ == "__main__":
    main()
