"""
Crop the top rows off a staged DrivingStereo copy, IN PLACE, identically for
left, right and GT disparity (sky is not needed for this task).

At the default crop this turns 881x400 into 881x320. Train afterwards with
--image_size 288 720.

DESTRUCTIVE: this rewrites the files it is pointed at. Point it at a staging
copy, never at the original download. --root is required rather than defaulted
for that reason, and --dry-run reports what would change without writing.

Usage:
    python tools/crop_dataset.py --root C:/ds_stage/DrivingStereo --dry-run
    python tools/crop_dataset.py --root C:/ds_stage/DrivingStereo
"""

import argparse
import glob
import os
import sys

import cv2


def crop_set(pattern, flags, label, crop_top, dry_run):
    """Crop every file matching `pattern`, rewriting it in place."""
    files = glob.glob(pattern)
    done = 0
    for path in files:
        img = cv2.imread(path, flags)
        if img is None or img.shape[0] <= crop_top:
            continue
        if not dry_run:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.endswith(".jpg") else []
            cv2.imwrite(path, img[crop_top:], params)
        done += 1
    verb = "would crop" if dry_run else "cropped"
    print(f"[{label}] {verb} {done}/{len(files)}")
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Crop the sky off a staged DrivingStereo copy, in place.")
    parser.add_argument("--root", required=True,
                        help="staged DrivingStereo root containing left/, "
                             "right/ and disparity/. This copy is modified.")
    parser.add_argument("--crop-top", type=int, default=80,
                        help="rows to remove from the top (default 80)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Not a directory: {args.root}")

    if args.dry_run:
        print("[dry-run] no files will be modified")

    crop_set(os.path.join(args.root, "left", "*", "*.jpg"),
             cv2.IMREAD_COLOR, "left ", args.crop_top, args.dry_run)
    crop_set(os.path.join(args.root, "right", "*", "*.jpg"),
             cv2.IMREAD_COLOR, "right", args.crop_top, args.dry_run)
    # GT is 16-bit PNG, so IMREAD_UNCHANGED preserves the disparity*256 values.
    crop_set(os.path.join(args.root, "disparity", "*", "*.png"),
             cv2.IMREAD_UNCHANGED, "disp ", args.crop_top, args.dry_run)

    samples = glob.glob(os.path.join(args.root, "left", "*", "*.jpg"))
    if samples:
        height, width = cv2.imread(samples[0]).shape[:2]
        print(f"[check] sample left image is now {width}x{height}")


if __name__ == "__main__":
    main()
