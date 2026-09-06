import sys
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_cloud.py path/to/cloud.ply")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(str(path))
    print(f"Loaded {len(pcd.points):,} points from {path.name}")

    if len(pcd.points) == 0:
        print("Empty point cloud!")
        sys.exit(1)

    pts = np.asarray(pcd.points)
    print(f"X range: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}] m")
    print(f"Y range: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}] m")
    print(f"Z range: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}] m")
    print(f"Center: ({pts.mean(axis=0)})")


    frame_size = max(pts[:, 2].max() / 10, 0.5)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=[0, 0, 0]
    )

    print("\nControls:")
    print("  Left-drag  : rotate")
    print("  Right-drag : pan")
    print("  Scroll     : zoom")
    print("  R          : reset view")
    print("  Q          : quit")
    print("\nAxes: Red=X (right), Green=Y (down), Blue=Z (forward)")

    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name=f"Point Cloud: {path.name}",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()