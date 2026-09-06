"""
Clean a stereo point cloud for presentation.

Stereo clouds have floating 'speckle' at object edges and low-texture regions
(building faces, tree edges) where matching fails. Statistical outlier removal
deletes points whose neighbours are far away, stripping the floating junk while
keeping the dense road/structure surfaces.

Usage:
    python clean_cloud.py input.ply output.ply
"""

import sys
import open3d as o3d
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python clean_cloud.py input.ply output.ply")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]

pcd = o3d.io.read_point_cloud(inp)
print(f"[load]  {len(pcd.points):,} points")

# 1) statistical outlier removal: drop points whose avg distance to their
#    `nb_neighbors` nearest points is more than `std_ratio` stds above the mean.
#    Lower std_ratio = more aggressive. 2.0 is gentle, 1.0 is strong.
clean, kept = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"[outlier] kept {len(kept):,} / {len(pcd.points):,} "
      f"(removed {len(pcd.points) - len(kept):,} floaters)")

# 2) (optional) radius outlier removal — catches isolated blobs the stat filter
#    misses. Comment out if it removes too much.
clean, kept2 = clean.remove_radius_outlier(nb_points=8, radius=0.5)
print(f"[radius]  kept {len(kept2):,} after radius filter")

# 3) (optional) light voxel downsample to even out density for a tidy render.
#    Comment out to keep full resolution.
# clean = clean.voxel_down_sample(voxel_size=0.05)
# print(f"[voxel]   {len(clean.points):,} points after 5cm downsample")

o3d.io.write_point_cloud(outp, clean)
print(f"[save]  -> {outp}  ({len(clean.points):,} points)")
