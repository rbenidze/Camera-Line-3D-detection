# Annotation guide

STUB. Not yet written.

Nothing in the repository or the surrounding files records the annotation
conventions, so this cannot be reconstructed from what is on disk. It has to be
written from memory of how the labelling was actually done.

## What it needs to cover

**The two classes.** `painted_line` and `track_edge`. What separates them, and
what to do at the boundary case where a painted line sits directly on a physical
edge.

**Polygon conventions.** Instance segmentation polygons, not boxes. How far down
the image a boundary is traced, where a line is cut when it runs out of the
frame, and whether occluded runs are split into separate instances or bridged.

**Frame selection.** Frames were sampled sparsely because consecutive frames are
near duplicates. `tools/sample_frames.py --every 30` is the operation that
produced the annotation set. Record which sampling was actually used for the
final dataset.

**Dataset split.** 1344 training, 112 validation, 52 test images. Record how the
split was made, in particular whether near-duplicate frames were kept out of
opposite sides of it, since that is what makes the held-out numbers meaningful.

**Model versions.** The detection scripts referenced Roboflow model versions 1,
2 and 3, and two different class naming schemes (`painted_line`/`track_edge` and
`line`/`edge`). Record which version produced the reported 88.0% precision,
84.7% recall and 73% mAP, and which naming scheme that version uses.
