# EXPERIMENT: classical CV track-edge detection with LiDAR projection, the approach used before RF-DETR. RESULT: superseded by the learned detector.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

mcap_path = r"C:\Users\haise\Documents\intern_practice_ds.mcap"
camera_topic = "/sensor/camera_fl/compressed_image"
lidar_topic = "/sensor/lidar_front/points"

typestore = get_typestore(Stores.ROS2_HUMBLE)

K = np.array([
    [860, 0, 640],
    [0, 860, 420],
    [0,   0,   1]
], dtype=np.float64)

t_cam = np.array([0.946467741935484, 0.16, 0.344])
r_cam = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
t_lid = np.array([0.951467741935484, 0.0, 0.346])
r_lid = Rotation.from_quat([9.44719270589705e-05, -0.01111577121396888, -9.494350635749763e-07, 0.9999382134434469])

# --- Step 1: Read camera + lidar ---
camera_image = None
lidar_points = None

with open(mcap_path, "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    schema_map = summary.schemas

    for schema, channel, message in reader.iter_messages():
        if channel.topic == camera_topic and camera_image is None:
            schema = schema_map[channel.schema_id]
            msg = typestore.deserialize_cdr(message.data, schema.name)
            compressed = np.frombuffer(msg.data, dtype=np.uint8)
            camera_image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

        elif channel.topic == lidar_topic and lidar_points is None:
            schema = schema_map[channel.schema_id]
            msg = typestore.deserialize_cdr(message.data, schema.name)
            dtype = np.dtype([
                ("x", np.float32), ("y", np.float32), ("z", np.float32),
                ("timestamp", np.float32), ("intensity", np.float32),
            ])
            arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
            lidar_points = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)

        if camera_image is not None and lidar_points is not None:
            break

# --- Step 2: Transform to base_link ---
points_base = (r_lid.as_matrix() @ lidar_points.T).T + t_lid

print(f"Z range in base_link: {points_base[:, 2].min():.2f} to {points_base[:, 2].max():.2f}")

# --- Step 3: Find LEFT track border (reliable side) ---
forward_mask = points_base[:, 0] > 2.0
forward_points = points_base[forward_mask]

ground_z = -0.45
height_jump_threshold = 0.10

left_border_raw = []
right_border_raw = []
slice_width = 1.5
y_bin_size = 0.3

for x_start in np.arange(3.0, 80.0, slice_width):
    x_end = x_start + slice_width
    slice_mask = (forward_points[:, 0] >= x_start) & (forward_points[:, 0] < x_end)
    slice_pts = forward_points[slice_mask]

    if len(slice_pts) < 20:
        continue

    median_x = np.median(slice_pts[:, 0])

    # Scan from center to LEFT
    left_edge_y = None
    for y_start in np.arange(0, 15, y_bin_size):
        y_end = y_start + y_bin_size
        bin_mask = (slice_pts[:, 1] >= y_start) & (slice_pts[:, 1] < y_end)
        bin_pts = slice_pts[bin_mask]

        if len(bin_pts) < 2:
            if y_start > 1.0:
                left_edge_y = y_start
            break

        min_z = np.min(bin_pts[:, 2])
        if min_z > ground_z + height_jump_threshold:
            left_edge_y = y_start
            break

    # Scan from center to RIGHT
    right_edge_y = None
    for y_start in np.arange(0, -15, -y_bin_size):
        y_end = y_start - y_bin_size
        bin_mask = (slice_pts[:, 1] <= y_start) & (slice_pts[:, 1] > y_end)
        bin_pts = slice_pts[bin_mask]

        if len(bin_pts) < 2:
            if y_start < -1.0:
                right_edge_y = y_start
            break

        min_z = np.min(bin_pts[:, 2])
        if min_z > ground_z + height_jump_threshold:
            right_edge_y = y_start
            break

    if left_edge_y is not None:
        left_border_raw.append([median_x, left_edge_y, ground_z])
    if right_edge_y is not None:
        right_border_raw.append([median_x, right_edge_y, ground_z])

print(f"Left raw points: {len(left_border_raw)}")
print(f"Right raw points: {len(right_border_raw)}")

# --- Smooth left border ---
def smooth_border(border_raw, order=3):
    if len(border_raw) < 5:
        return np.array(border_raw) if len(border_raw) > 0 else np.empty((0, 3))

    border_raw = np.array(border_raw)
    x = border_raw[:, 0]
    y = border_raw[:, 1]

    coeffs = np.polyfit(x, y, order)
    y_fit = np.polyval(coeffs, x)

    residuals = np.abs(y - y_fit)
    median_res = np.median(residuals)
    inlier_mask = residuals < max(median_res * 2.5, 0.3)

    if np.sum(inlier_mask) > order + 1:
        coeffs = np.polyfit(x[inlier_mask], y[inlier_mask], order)

    x_smooth = np.arange(x.min(), x.max(), 1.0)
    y_smooth = np.polyval(coeffs, x_smooth)
    z_smooth = np.full_like(x_smooth, ground_z)

    return np.column_stack([x_smooth, y_smooth, z_smooth])

left_border = smooth_border(left_border_raw)

# --- Generate right border by mirroring left border ---
# Estimate track width from the few good right border points
# From BEV: left is at Y ≈ +3, right should be at Y ≈ -7 to -5
# So track width ≈ 8-10m. Let's measure from the raw points we have.
track_width = 10.0  # default

if len(right_border_raw) > 3 and len(left_border_raw) > 3:
    left_raw = np.array(left_border_raw)
    right_raw = np.array(right_border_raw)
    
    # Calculate width from matching X positions
    widths = []
    for rpt in right_raw:
        x_diffs = np.abs(left_raw[:, 0] - rpt[0])
        nearest_idx = np.argmin(x_diffs)
        if x_diffs[nearest_idx] < slice_width * 2:
            w = left_raw[nearest_idx, 1] - rpt[1]
            if 5.0 < w < 15.0:
                widths.append(w)
    
    if len(widths) > 0:
        track_width = np.median(widths)

print(f"Estimated track width: {track_width:.1f}m")

# Mirror: right border = left border shifted by track width
right_border = left_border.copy()
right_border[:, 1] = left_border[:, 1] - track_width

print(f"Left border points (smoothed): {len(left_border)}")
print(f"Right border points (mirrored): {len(right_border)}")

# --- Step 4: Project borders onto camera image ---
def project_to_image(points_3d, t_cam, r_cam, K):
    if len(points_3d) == 0:
        return np.array([]), np.array([]), np.array([]), np.empty((0, 3))

    points_cam_frame = (r_cam.inv().as_matrix() @ (points_3d - t_cam).T).T

    points_camera = np.zeros_like(points_cam_frame)
    points_camera[:, 0] = -points_cam_frame[:, 1]
    points_camera[:, 1] = -points_cam_frame[:, 2]
    points_camera[:, 2] = points_cam_frame[:, 0]

    mask = points_camera[:, 2] > 1.0
    points_camera = points_camera[mask]
    points_3d_filtered = points_3d[mask]

    points_2d = (K @ points_camera.T).T
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]

    u = points_2d[:, 0].astype(int)
    v = points_2d[:, 1].astype(int)
    depth = points_camera[:, 2]

    valid = (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
    return u[valid], v[valid], depth[valid], points_3d_filtered[valid]

output_image = camera_image.copy()

# Project left border
lu, lv, ldepth, left_3d = project_to_image(left_border, t_cam, r_cam, K)
print(f"Left border on image: {len(lu)} points")

for i in range(len(lu)):
    ratio = min(ldepth[i] / 80.0, 1.0)
    r = int(255 * (1 - ratio))
    b = int(255 * ratio)
    cv2.circle(output_image, (lu[i], lv[i]), 5, (b, 0, r), -1)

# Project right border
ru, rv, rdepth, right_3d = project_to_image(right_border, t_cam, r_cam, K)
print(f"Right border on image: {len(ru)} points")

for i in range(len(ru)):
    ratio = min(rdepth[i] / 80.0, 1.0)
    r = int(255 * (1 - ratio))
    b = int(255 * ratio)
    cv2.circle(output_image, (ru[i], rv[i]), 5, (b, 0, r), -1)

# Draw smooth lines
if len(lu) > 1:
    pts_left = np.column_stack((lu, lv))
    for i in range(len(pts_left) - 1):
        ratio = min(((ldepth[i] + ldepth[i+1]) / 2) / 80.0, 1.0)
        r = int(255 * (1 - ratio))
        b = int(255 * ratio)
        cv2.line(output_image, tuple(pts_left[i]), tuple(pts_left[i+1]), (b, 0, r), 3)

if len(ru) > 1:
    pts_right = np.column_stack((ru, rv))
    for i in range(len(pts_right) - 1):
        ratio = min(((rdepth[i] + rdepth[i+1]) / 2) / 80.0, 1.0)
        r = int(255 * (1 - ratio))
        b = int(255 * ratio)
        cv2.line(output_image, tuple(pts_right[i]), tuple(pts_right[i+1]), (b, 0, r), 3)

cv2.imwrite("track_detection_result.png", output_image)
print("Saved track_detection_result.png")

# --- Step 5: Bird's eye view ---
plt.figure(figsize=(12, 8))
low_pts = forward_points[forward_points[:, 2] < -0.1]
plt.scatter(low_pts[::5, 0], low_pts[::5, 1], c='gray', s=0.5, alpha=0.3, label='Ground')
high_pts = forward_points[forward_points[:, 2] >= -0.1]
plt.scatter(high_pts[::5, 0], high_pts[::5, 1], c='orange', s=0.5, alpha=0.3, label='Above ground')
plt.scatter(left_border[:, 0], left_border[:, 1], c='red', s=10, zorder=5, label='Left border')
plt.scatter(right_border[:, 0], right_border[:, 1], c='blue', s=10, zorder=5, label='Right border')
if len(left_border_raw) > 0:
    raw_l = np.array(left_border_raw)
    plt.scatter(raw_l[:, 0], raw_l[:, 1], c='red', s=50, marker='x', zorder=6, label='Left raw')
if len(right_border_raw) > 0:
    raw_r = np.array(right_border_raw)
    plt.scatter(raw_r[:, 0], raw_r[:, 1], c='blue', s=50, marker='x', zorder=6, label='Right raw')
plt.xlabel('X (forward, meters)')
plt.ylabel('Y (left/right, meters)')
plt.title(f'Track Borders - Bird Eye View (width={track_width:.1f}m)')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.savefig("track_borders_bev.png", dpi=150)
print("Saved track_borders_bev.png")
plt.show()

# Save 3D border points
if len(left_3d) > 0 and len(right_3d) > 0:
    np.save("left_border_3d.npy", left_3d)
    np.save("right_border_3d.npy", right_3d)
    print(f"Saved left_border_3d.npy ({left_3d.shape}) and right_border_3d.npy ({right_3d.shape})")