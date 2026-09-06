"""
Dump the sensor extrinsics from an MCAP recording's /tf and /tf_static topics.

Merges read_tf.py and list_camera_topics.py. The latter was misnamed: it dumped
the first /tf_static message, it did not list camera topics.

These transforms are what tools/project_lidar.py uses to put LiDAR points into
the camera frame. The camera_fl mounting rotation recovered here is the fix that
made the LiDAR overlay land on real geometry.

Usage:
    python tools/read_transforms.py --mcap run.mcap
    python tools/read_transforms.py --mcap run.mcap --static
"""

import argparse
from pathlib import Path

from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore


def read_dynamic(reader, typestore, max_messages):
    """Collect unique parent -> child transforms from /tf."""
    schema_map = reader.get_summary().schemas
    transforms = {}
    seen = 0

    for _schema, channel, message in reader.iter_messages():
        if channel.topic != "/tf":
            continue
        schema = schema_map[channel.schema_id]
        msg = typestore.deserialize_cdr(message.data, schema.name)
        for tf in msg.transforms:
            translation = tf.transform.translation
            rotation = tf.transform.rotation
            transforms[(tf.header.frame_id, tf.child_frame_id)] = (
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w),
            )
        seen += 1
        if seen > max_messages:
            break

    print("=== transforms (parent -> child) ===")
    for (parent, child), (t, q) in sorted(transforms.items()):
        print(f"{parent} -> {child}")
        print(f"   t=[{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]  "
              f"q=[{q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}]")


def read_static(reader, typestore):
    """Print the first /tf_static message verbatim."""
    schema_map = reader.get_summary().schemas
    for _schema, channel, message in reader.iter_messages():
        if channel.topic != "/tf_static":
            continue
        schema = schema_map[channel.schema_id]
        print(typestore.deserialize_cdr(message.data, schema.name))
        return
    print("no /tf_static messages found")


def main():
    parser = argparse.ArgumentParser(
        description="Dump sensor extrinsics from an MCAP recording.")
    parser.add_argument("--mcap", type=Path, required=True,
                        help="path to the .mcap recording")
    parser.add_argument("--static", action="store_true",
                        help="print the first /tf_static message instead of /tf")
    parser.add_argument("--max-messages", type=int, default=200,
                        help="how many /tf messages to scan (default 200)")
    args = parser.parse_args()

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        if args.static:
            read_static(reader, typestore)
        else:
            read_dynamic(reader, typestore, args.max_messages)


if __name__ == "__main__":
    main()
