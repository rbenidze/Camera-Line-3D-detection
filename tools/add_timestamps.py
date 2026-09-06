"""
Add per-frame capture timestamps to the Yas Marina stereo-pair manifest.

WHY THIS EXISTS
---------------
tools/build_lidar_gt.py needs to match each stereo pair to the LiDAR sweep
nearest in TIME. The sweeps are already named by message.log_time (see
tools/extract_lidar.py). But the camera frames were written as sequential names
(frame_000000.png ...) by tools/extract_frames.py, so their capture time was
lost. Index-based frame to sweep pairing is WRONG because the camera and LiDAR
topics publish at different rates; it produced a 5x spread in the recovered
fx*B (2292 / 1711 / 8425 / 3528) instead of a tight cluster.

This script replays the MCAP camera topic with the IDENTICAL logic
tools/extract_frames.py uses (same topic, same decode, increment only on
successful decode), so the Nth successfully decoded message here corresponds to
frame_{N:06d}.png exactly. For each frame it records message.log_time, the same
clock the sweep filenames use, then writes a "timestamp" field onto every pair
in manifest.json, keyed by frame_index.

Afterwards re-run the fx*B recovery. The values should collapse to a tight
cluster if the projection is sound.

Usage:
    python tools/add_timestamps.py --mcap run.mcap \
        --manifest ~/RAFT-Stereo/datasets/YasMarina/training/manifest.json
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore


def needed_frame_indices(manifest):
    """The frame_index values we actually need timestamps for."""
    return {int(p["frame_index"]) for p in manifest["pairs"]}


def collect_frame_timestamps(mcap_path, camera_topic, wanted):
    """
    Replay the camera topic EXACTLY like tools/extract_frames.py and return
    {frame_index: log_time_ns} for every frame_index in `wanted`.

    Critical detail: the extractor increments its counter ONLY after a
    successful cv2.imdecode, continuing on failure. We replicate that so this
    numbering matches the saved filenames one for one even if some messages
    failed to decode.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    out = {}
    max_wanted = max(wanted) if wanted else -1

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None:
            raise RuntimeError("No MCAP summary found.")
        schema_map = summary.schemas

        frame_count, decoded_ok, decode_fail = 0, 0, 0
        for _schema, channel, message in reader.iter_messages():
            if channel.topic != camera_topic:
                continue
            schema = schema_map[channel.schema_id]
            msg = typestore.deserialize_cdr(message.data, schema.name)
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                decode_fail += 1
                continue  # mirror the extractor: do NOT advance frame_count
            if frame_count in wanted:
                out[frame_count] = int(message.log_time)
            frame_count += 1
            decoded_ok += 1
            if frame_count > max_wanted and len(out) == len(wanted):
                break

        print(f"[mcap] camera messages decoded ok: {decoded_ok}, "
              f"decode failures (skipped): {decode_fail}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Write camera capture timestamps into a stereo-pair manifest.")
    parser.add_argument("--mcap", type=Path, required=True,
                        help="path to the .mcap recording")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="manifest.json written by tools/sample_stereo_pairs.py")
    parser.add_argument("--camera-topic",
                        default="/sensor/camera_fl/compressed_image",
                        help="MUST match the topic used by tools/extract_frames.py")
    parser.add_argument("--force", action="store_true",
                        help="rewrite timestamps even if they are already present")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    if (manifest["pairs"] and not args.force
            and all("timestamp" in p for p in manifest["pairs"])):
        print("[ok] manifest already has timestamps on every pair. "
              "Nothing to do (use --force to rewrite).")
        return

    wanted = needed_frame_indices(manifest)
    if not wanted:
        raise SystemExit("Manifest contains no pairs.")
    print(f"[manifest] {len(manifest['pairs'])} pairs, "
          f"{len(wanted)} distinct frame indices needed "
          f"(min {min(wanted)}, max {max(wanted)})")

    timestamps = collect_frame_timestamps(args.mcap, args.camera_topic, wanted)

    missing = sorted(wanted - set(timestamps))
    if missing:
        print(f"[ERROR] no timestamp for {len(missing)} frame indices: "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
        print("        The MCAP may have fewer decodable camera frames than "
              "expected, or --camera-topic is wrong. Not writing manifest.")
        return

    for pair in manifest["pairs"]:
        pair["timestamp"] = timestamps[int(pair["frame_index"])]

    # Timestamps should increase with frame_index.
    seq = sorted((int(p["frame_index"]), int(p["timestamp"]))
                 for p in manifest["pairs"])
    non_mono = sum(1 for a, b in zip(seq, seq[1:]) if b[1] <= a[1])
    if non_mono:
        print(f"[warn] {non_mono} timestamp pairs are non-increasing with frame "
              "order; unexpected, inspect before trusting.")
    else:
        span_s = (seq[-1][1] - seq[0][1]) / 1e9
        print(f"[ok] timestamps monotonic with frame index; recording spans "
              f"{span_s:.1f} s across sampled frames")

    backup = args.manifest.with_suffix(args.manifest.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(args.manifest, backup)
        print(f"[backup] original manifest -> {backup}")
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote timestamps onto {len(manifest['pairs'])} pairs "
          f"in {args.manifest}")


if __name__ == "__main__":
    main()
