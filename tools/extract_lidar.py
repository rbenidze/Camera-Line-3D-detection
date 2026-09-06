"""
Decode LiDAR sweeps out of an MCAP recording into .npy point arrays.

From save_lidar_points.py. LiDAR is used only during development to build
ground truth for evaluating the depth stage. It is not used at inference.

Each sweep is saved as <log_time_ns>.npy holding an (N, 4) array of
x, y, z, intensity. Naming by timestamp rather than index is deliberate:
tools/add_timestamps.py and tools/build_lidar_gt.py match frames to sweeps by
time, because the camera and LiDAR topics publish at different rates.

Usage:
    python tools/extract_lidar.py --mcap run.mcap --out lidar_front
"""

import argparse
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

POINT_DTYPE = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("timestamp", np.float32),
    ("intensity", np.float32),
])


def main():
    parser = argparse.ArgumentParser(
        description="Extract LiDAR sweeps from an MCAP recording.")
    parser.add_argument("--mcap", type=Path, required=True,
                        help="path to the .mcap recording")
    parser.add_argument("--topic", default="/sensor/lidar_front/points",
                        help="LiDAR point cloud topic")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for .npy sweeps")
    parser.add_argument("--limit", type=int,
                        help="stop after this many sweeps")
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
            arr = np.frombuffer(msg.data, dtype=POINT_DTYPE,
                                count=msg.width * msg.height)
            points = np.stack([arr["x"], arr["y"], arr["z"], arr["intensity"]],
                              axis=1)

            np.save(args.out / f"{message.log_time}.npy", points)
            count += 1
            if count % 100 == 0:
                print(f"Saved {count} sweeps...")
            if args.limit and count >= args.limit:
                break

        print(f"Done. {count} LiDAR sweeps saved to {args.out}/")


if __name__ == "__main__":
    main()
