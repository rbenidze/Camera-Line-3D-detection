"""
Inspect an MCAP recording: list its topics, and optionally probe one image
message for resolution and encoding.

Merges list_mcap_topics.py and get_transforms.py. The latter was misnamed: it
never read a transform, it decoded the first camera message to report frame size
and format.

Usage:
    python tools/list_topics.py --mcap run.mcap
    python tools/list_topics.py --mcap run.mcap --filter camera
    python tools/list_topics.py --mcap run.mcap \
        --probe-image /sensor/camera_fl/compressed_image
"""

import argparse
from pathlib import Path

from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore


def list_topics(reader, name_filter):
    summary = reader.get_summary()
    if summary is None:
        print("No summary found")
        return
    for _channel_id, channel in sorted(summary.channels.items()):
        if name_filter and name_filter not in channel.topic:
            continue
        print(f"topic={channel.topic} | schema={channel.schema_id} "
              f"| encoding={channel.message_encoding}")


def probe_image(reader, topic):
    """Decode the first message on `topic` and report frame size and format."""
    import cv2
    import numpy as np

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    schema_map = reader.get_summary().schemas

    for _schema, channel, message in reader.iter_messages():
        if channel.topic != topic:
            continue
        schema = schema_map[channel.schema_id]
        msg = typestore.deserialize_cdr(message.data, schema.name)
        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            print(f"{topic}: first message failed to decode")
            return
        print(f"{topic}: {image.shape[1]}x{image.shape[0]}, "
              f"format={getattr(msg, 'format', 'unknown')}")
        return
    print(f"{topic}: no messages found")


def main():
    parser = argparse.ArgumentParser(description="Inspect an MCAP recording.")
    parser.add_argument("--mcap", type=Path, required=True,
                        help="path to the .mcap recording")
    parser.add_argument("--filter", help="only show topics containing this text")
    parser.add_argument("--probe-image",
                        help="decode the first message on this topic and report "
                             "its resolution and format")
    args = parser.parse_args()

    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        if args.probe_image:
            probe_image(reader, args.probe_image)
        else:
            list_topics(reader, args.filter)


if __name__ == "__main__":
    main()
