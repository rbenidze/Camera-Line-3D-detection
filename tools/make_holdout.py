"""
Carve a held-out evaluation split out of a DrivingStereo tree.

Consecutive DrivingStereo frames are near duplicates. If near-identical frames
land on both sides of the split, the evaluation overstates accuracy, so frames
are taken on a stride rather than at random.

The stride is FLAT across the sorted frame list of all sequences combined, not
per sequence. A per-sequence stride gives short sequences disproportionate
weight in the holdout.

Every held-out frame moves as a triple: left image, right image and GT
disparity. A frame missing any of the three is skipped and reported, never
half-moved.

Writes config/holdout_manifest.json naming exactly which frames were held out.
src/evaluate.py reads that file rather than re-deriving a stride, so the
evaluation set is a recorded fact rather than something recomputed from
whatever happens to be on disk.

This is NOT the same mechanism as the Yas holdout in tools/build_lidar_gt.py,
which strides over its own KITTI-layout manifest. The two manifests are not
interchangeable.

Usage:
    python tools/make_holdout.py --root datasets/DrivingStereo \
        --holdout datasets/DrivingStereo_holdout --every 400
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

LEFT_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_MANIFEST = Path("config/holdout_manifest.json")


def collect_frames(root: Path):
    """Every frame with all three components present, sorted across sequences.

    Returns a list of dicts with the sequence, stem and the three relative
    paths. Sorting is by (sequence, stem) so the order is deterministic and
    independent of filesystem enumeration order.
    """
    left_root = root / "left"
    if not left_root.is_dir():
        sys.exit(f"No left/ directory under {root}")

    frames, incomplete = [], []
    for seq_dir in sorted(p for p in left_root.iterdir() if p.is_dir()):
        seq = seq_dir.name
        for left in sorted(seq_dir.iterdir()):
            if left.suffix.lower() not in LEFT_SUFFIXES:
                continue
            stem = left.stem
            right = root / "right" / seq / left.name
            disp = root / "disparity" / seq / f"{stem}.png"
            if not right.exists() or not disp.exists():
                incomplete.append(f"{seq}/{stem}")
                continue
            frames.append({
                "sequence": seq,
                "stem": stem,
                "left": str(Path("left") / seq / left.name).replace("\\", "/"),
                "right": str(Path("right") / seq / left.name).replace("\\", "/"),
                "disparity": str(Path("disparity") / seq / f"{stem}.png").replace("\\", "/"),
            })
    return frames, incomplete


def main():
    parser = argparse.ArgumentParser(
        description="Move a strided holdout split out of a DrivingStereo tree.")
    parser.add_argument("--root", type=Path, required=True,
                        help="DrivingStereo root containing left/, right/, disparity/")
    parser.add_argument("--holdout", type=Path, required=True,
                        help="destination root for the held-out frames")
    parser.add_argument("--every", type=int, default=400,
                        help="take every Nth frame of the combined sorted list "
                             "(default 400)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"manifest to write (default {DEFAULT_MANIFEST})")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the split without moving anything")
    args = parser.parse_args()

    if args.every < 1:
        parser.error("--every must be at least 1")

    frames, incomplete = collect_frames(args.root)
    if not frames:
        sys.exit(f"No complete frame triples found under {args.root}")
    print(f"[scan] {len(frames)} complete triples across "
          f"{len({f['sequence'] for f in frames})} sequences")
    if incomplete:
        print(f"[scan] {len(incomplete)} frames skipped, missing right or "
              f"disparity (e.g. {', '.join(incomplete[:3])})")

    selected = frames[::args.every]
    print(f"[split] every {args.every} -> {len(selected)} holdout frames")

    if args.dry_run:
        for frame in selected[:10]:
            print(f"    {frame['sequence']}/{frame['stem']}")
        if len(selected) > 10:
            print(f"    ... and {len(selected) - 10} more")
        print("[dry-run] nothing moved, no manifest written")
        return

    moved = []
    for frame in selected:
        ok = True
        for key in ("left", "right", "disparity"):
            src = args.root / frame[key]
            dst = args.holdout / frame[key]
            if not src.exists():
                print(f"  [skip] {frame['sequence']}/{frame['stem']}: "
                      f"{key} vanished before the move")
                ok = False
                break
            dst.parent.mkdir(parents=True, exist_ok=True)
        if not ok:
            continue
        for key in ("left", "right", "disparity"):
            shutil.move(str(args.root / frame[key]),
                        str(args.holdout / frame[key]))
        moved.append(frame)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(args.root),
        "holdout_root": str(args.holdout),
        "every": args.every,
        "total_frames_scanned": len(frames),
        "holdout_count": len(moved),
        "note": ("Flat stride across the sorted frame list of all sequences "
                 "combined. Paths are relative to holdout_root."),
        "frames": moved,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"[move] {len(moved)} triples -> {args.holdout}")
    print(f"[manifest] wrote {args.manifest}")
    print("[next] evaluate with:")
    print(f"    python src/evaluate.py --holdout-root {args.holdout} \\")
    print(f"        --manifest {args.manifest} --checkpoint <weights> \\")
    print("        --calib config/stereo_drivingstereo.json")


if __name__ == "__main__":
    main()
