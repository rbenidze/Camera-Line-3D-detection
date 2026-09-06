"""
Sample and crop Yas Marina stereo pairs into KITTI-2015 layout.

Step 1 of RAFT-Stereo retraining:

- Samples every Nth left/right pair, so near-duplicate consecutive frames do
  not dominate the set.
- Applies a FIXED BAND CROP, IDENTICAL on left and right. This is what
  preserves disparity: the same window in both images leaves the horizontal
  shift unchanged. Never crop the two sides differently.
- Saves as:
      <dst>/image_2/000123_10.png   (left)
      <dst>/image_3/000123_10.png   (right)
      <dst>/disp_occ_0/             (filled later by tools/build_lidar_gt.py)
- Writes manifest.json recording the crop and the frame mapping.
  tools/build_lidar_gt.py MUST read it so its crop matches exactly.

After running, open the preview image and check that the car nose is mostly
gone at the bottom, track lines and structures are kept, and excess sky is gone
at the top. Adjust --crop-top / --crop-bottom and re-run if needed.

Usage:
    python tools/sample_stereo_pairs.py \
        --src-left camera_frames --src-right camera_frames_fr \
        --dst ~/RAFT-Stereo/datasets/YasMarina/training \
        --every 25 --crop-top 180 --crop-bottom 720
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Sample and crop stereo pairs into KITTI-2015 layout.")
    parser.add_argument("--src-left", type=Path, required=True,
                        help="left (fl) frame directory")
    parser.add_argument("--src-right", type=Path, required=True,
                        help="right (fr) frame directory")
    parser.add_argument("--dst", type=Path, required=True,
                        help="output training directory")
    parser.add_argument("--every", type=int, default=25,
                        help="sample every Nth pair (default 25)")
    parser.add_argument("--crop-top", type=int, default=180,
                        help="first row kept (default 180)")
    parser.add_argument("--crop-bottom", type=int, default=720,
                        help="last row kept, exclusive (default 720)")
    parser.add_argument("--pattern", default="frame_*.png",
                        help="left frame glob (default frame_*.png)")
    args = parser.parse_args()

    if args.crop_bottom <= args.crop_top:
        parser.error("--crop-bottom must be greater than --crop-top")

    image_2 = args.dst / "image_2"
    image_3 = args.dst / "image_3"
    image_2.mkdir(parents=True, exist_ok=True)
    image_3.mkdir(parents=True, exist_ok=True)
    (args.dst / "disp_occ_0").mkdir(parents=True, exist_ok=True)

    lefts = sorted(glob.glob(str(args.src_left / args.pattern)))
    print(f"[src] {len(lefts)} left frames")
    if not lefts:
        raise SystemExit(f"No frames matching {args.pattern} in {args.src_left}")

    manifest = {
        "crop_top": args.crop_top,
        "crop_bottom": args.crop_bottom,
        "every_n": args.every,
        "src_left": str(args.src_left),
        "src_right": str(args.src_right),
        "pairs": [],
    }

    kept, missing, preview_done = 0, 0, False

    for left_path in lefts[::args.every]:
        base = os.path.basename(left_path)
        frame_idx = int(base.split("_")[1].split(".")[0])
        right_path = args.src_right / base
        if not right_path.exists():
            missing += 1
            continue

        left = cv2.imread(left_path)
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            missing += 1
            continue

        # Identical band crop on BOTH images.
        left_crop = left[args.crop_top:args.crop_bottom, :]
        right_crop = right[args.crop_top:args.crop_bottom, :]

        kitti_name = f"{kept:06d}_10.png"
        cv2.imwrite(str(image_2 / kitti_name), left_crop)
        cv2.imwrite(str(image_3 / kitti_name), right_crop)
        manifest["pairs"].append({"kitti_name": kitti_name,
                                  "frame_index": frame_idx})

        if not preview_done:
            cv2.imwrite(str(args.dst / "crop_preview_first_pair.png"),
                        cv2.vconcat([left_crop, right_crop]))
            preview_done = True

        kept += 1

    with open(args.dst / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    height = args.crop_bottom - args.crop_top
    print(f"[done] {kept} pairs saved at height {height} "
          f"({missing} skipped, missing or unreadable)")
    print(f"[out]  {args.dst}")
    print("[next] open crop_preview_first_pair.png and check the band, "
          "then generate GT with tools/build_lidar_gt.py")


if __name__ == "__main__":
    main()
