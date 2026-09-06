"""
Copy every Nth frame out of a frame directory, for annotation upload.

Merges sample_frames.py, select_frames.py and select_frames_every_30_from_3200.py,
which were three copies of the same operation differing only in N, start index
and whether they pulled from one directory or two.

Consecutive frames are near duplicates, so sampling sparsely is what gives the
annotation set distinct scenes. For detection training we use left frames only.

Usage:
    # every 30th left frame
    python tools/sample_frames.py --src camera_frames --dst frames_to_label --every 30

    # every 30th frame from index 3200 on
    python tools/sample_frames.py --src camera_frames --dst frames_to_label \
        --every 30 --start 3200

    # matched left/right pairs, prefixed so they stay distinguishable
    python tools/sample_frames.py --src camera_frames --src-right camera_frames_fr \
        --dst selected_frames --every 50
"""

import argparse
import shutil
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def frames_in(directory: Path):
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in IMAGE_SUFFIXES)


def main():
    parser = argparse.ArgumentParser(
        description="Sample every Nth frame into an output directory.")
    parser.add_argument("--src", type=Path, required=True,
                        help="source frame directory (left camera)")
    parser.add_argument("--src-right", type=Path,
                        help="optional right-camera directory; matching frames "
                             "are copied alongside with l_/r_ prefixes")
    parser.add_argument("--dst", type=Path, required=True,
                        help="output directory")
    parser.add_argument("--every", type=int, default=30,
                        help="sampling interval (default 30)")
    parser.add_argument("--start", type=int, default=0,
                        help="index to start from (default 0)")
    parser.add_argument("--limit", type=int,
                        help="stop after copying this many frames")
    args = parser.parse_args()

    frames = frames_in(args.src)[args.start::args.every]
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"No frames selected from {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    copied = 0

    for frame in frames:
        if args.src_right is None:
            shutil.copy2(frame, args.dst / frame.name)
            copied += 1
            continue

        right = args.src_right / frame.name
        if not right.exists():
            print(f"skipped {frame.name}: no matching right frame")
            continue
        shutil.copy2(frame, args.dst / f"l_{frame.name}")
        shutil.copy2(right, args.dst / f"r_{right.name}")
        copied += 2

    print(f"Copied {copied} file(s) from {len(frames)} sampled frame(s) "
          f"to {args.dst}/")


if __name__ == "__main__":
    main()
