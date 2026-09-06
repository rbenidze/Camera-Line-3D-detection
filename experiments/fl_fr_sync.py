# EXPERIMENT: check whether fl and fr frames are time-synced when paired by index. RESULT: they are not. The offset varies per frame, 78 px on frame 20 against 300 px on frame 200.
#
# Kept as a record of the attempt. Not part of the pipeline and not cleaned up.

"""
Check whether the LEFT (fl) and RIGHT (fr) cameras are time-synced per frame.

The stereo pairs were sampled by INDEX: frame_000200.png from camera_frames (fl)
paired with frame_000200.png from camera_frames_fr (fr). But fl and fr are
independent topics. If they publish at different times, "pair 200" is two views
taken at different moments -> the car moved between them -> an apparent horizontal
offset that VARIES per frame. That matches the offset-fit result (C=78px on frame
20, C=300px on frame 200): a per-frame-varying offset, not a fixed rectification
constant.

This reads both camera topics' log_time in publish order (mirroring
decode_camera_image.py) and reports, for the sampled frame indices, the time gap
between the Nth fl frame and the Nth fr frame.

Place in C:\\Users\\haise\\Documents\\Camera_line and run:
    python fl_fr_sync.py
"""
import os, json
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore
import cv2

MCAP = r"C:\Users\haise\Documents\intern_practice_ds.mcap"
FL_TOPIC = "/sensor/camera_fl/compressed_image"
FR_TOPIC = "/sensor/camera_fr/compressed_image"
MANIFEST = r"C:\Users\haise\RAFT-Stereo\datasets\YasMarina\training\manifest.json"

def collect_times(topic, max_idx):
    """log_time per successfully-decoded frame, mirroring decode_camera_image.py."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    out = []
    with open(MCAP, "rb") as f:
        reader = make_reader(f)
        smap = reader.get_summary().schemas
        for schema, channel, message in reader.iter_messages():
            if channel.topic != topic:
                continue
            sch = smap[channel.schema_id]
            msg = ts.deserialize_cdr(message.data, sch.name)
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out.append(int(message.log_time))
            if len(out) > max_idx:
                break
    return out

def main():
    mf = json.load(open(MANIFEST))
    idxs = sorted({int(p["frame_index"]) for p in mf["pairs"]})
    max_idx = max(idxs)
    print("[read] fl timestamps...")
    fl = collect_times(FL_TOPIC, max_idx)
    print("[read] fr timestamps...")
    fr = collect_times(FR_TOPIC, max_idx)
    print("fl frames: %d   fr frames: %d" % (len(fl), len(fr)))

    print("\nframe_idx   fl_time-fr_time (ms)   (index-paired gap)")
    gaps = []
    for i in idxs:
        if i < len(fl) and i < len(fr):
            g = (fl[i] - fr[i]) / 1e6
            gaps.append(g)
    gaps = np.array(gaps)
    # show a sample
    for i in [idxs[0], idxs[len(idxs)//4], idxs[len(idxs)//2], idxs[3*len(idxs)//4], idxs[-1]]:
        if i < len(fl) and i < len(fr):
            print("  %6d        %8.1f ms" % (i, (fl[i]-fr[i])/1e6))
    print("\nindex-paired fl-fr gap:  mean=%.1f ms  std=%.1f ms  min=%.1f  max=%.1f"
          % (gaps.mean(), gaps.std(), gaps.min(), gaps.max()))

    # what if we instead match each fl frame to the NEAREST fr frame in time?
    fr_arr = np.array(fr)
    best_gaps = []
    for i in idxs:
        if i < len(fl):
            j = np.argmin(np.abs(fr_arr - fl[i]))
            best_gaps.append((fl[i] - fr_arr[j]) / 1e6)
    best_gaps = np.array(best_gaps)
    print("nearest-time fl->fr gap: mean=%.1f ms  std=%.1f ms  max=%.1f ms"
          % (np.abs(best_gaps).mean(), best_gaps.std(), np.abs(best_gaps).max()))
    print("\nIf index-paired gap is large/variable but nearest-time gap is small,")
    print("the pairs are desynced and should be re-sampled by timestamp.")

if __name__ == "__main__":
    main()
