"""
Decode compressed camera images out of an MCAP recording into PNG frames.

Merges decode_camera_image.py and right_camera.py, which were the same script
differing only in topic and output directory.

Frames are written as frame_%06d.png in publish order. Note that this discards
capture time, which is why tools/add_timestamps.py exists: pairing frames to
LiDAR sweeps by index is wrong because the topics publish at different rates.

Usage:
    python tools/extract_frames.py --mcap run.mcap \
        --topic /sensor/camera_fl/compressed_image --out camera_frames

    python tools/extract_frames.py --mcap run.mcap \
        --topic /sensor/camera_fr/compressed_image --out camera_frames_fr
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera frames from an MCAP recording.")
    parser.add_argument("--mcap", type=Path, required=True,
                        help="path to the .mcap recording")
    parser.add_argument("--topic", default="/sensor/camera_fl/compressed_image",
                        help="camera topic (default: front-left)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for PNG frames")
    parser.add_argument("--limit", type=int,
                        help="stop after this many frames")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None:
            raise RuntimeError("No MCAP summary found.")
        schema_map = summary.schemas

        count = 0
        for _schema, channel, message in reader.iter_messages():
            if channel.topic != args.topic:
                continue

            schema = schema_map[channel.schema_id]
            msg = typestore.deserialize_cdr(message.data, schema.name)
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                print(f"Frame {count}: failed to decode")
                continue

            cv2.imwrite(str(args.out / f"frame_{count:06d}.png"), image)
            count += 1
            if count % 100 == 0:
                print(f"Saved {count} frames...")
            if args.limit and count >= args.limit:
                break

        print(f"Done. {count} frames saved to {args.out}/")


if __name__ == "__main__":
    main()
