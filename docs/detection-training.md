# RF-DETR detection: dataset and training

There is no training configuration artifact in this repository. Training ran in
Roboflow's hosted interface, so nothing was ever written to disk. This document
reconstructs the process from the thesis, which is the only surviving record.

Everything below is sourced from the thesis, Section 4.1. Where a figure could
not be verified there, it is marked.

## Task

Instance segmentation, two boundary classes:

- `painted_line`, painted track markings
- `track_edge`, physical boundaries such as curbs and barriers

The detector output decides which pixels reach the 3D reconstruction stage, so
detection failure is pipeline failure.

## Dataset

Annotated frames from the Yas Marina recording, 1508 images total:

| Split | Images |
|---|---|
| Training | 1344 |
| Validation | 112 |
| Test | 52 |

The test set is held out, and evaluation uses a 50% confidence threshold.

Frames were sampled sparsely from the left camera stream rather than taken
consecutively. Consecutive video frames are near duplicates, and the left and
right streams of a stereo pair are near duplicates of each other, so dense
sampling would have inflated apparent performance by putting near-identical
images on both sides of the split. `tools/sample_frames.py` is the operation
that produced the annotation set.

## Results on the held-out test set

| Metric | Value |
|---|---|
| mAP@50 | 70.2% |
| Precision | 82.6% |
| Recall | 69.6% |
| F1 | 75.2% |

Per-class AP@50:

| Class | AP@50 |
|---|---|
| `painted_line` | 62.0% |
| `track_edge` | 79.0% |

Per-class precision, recall and F1 were not available in the Roboflow summary.

The per-class gap is the informative part. `track_edge` scores 17 points higher
than `painted_line`. Physical edges with curbs are large objects; painted lines
are thin and sometimes fragmented, so a small mask shift costs a large fraction
of the overlap-based score. This is the same thin-structure difficulty that
appears in the disparity stage, where edge EPE exceeds flat EPE, and in the 3D
stage, where high-gradient error is 39.5 cm against 21.9 cm overall. See
[experiments.md](experiments.md).

Precision exceeds recall by 13 points. The detector is more reliable when it
predicts a boundary than it is at finding every boundary. For the reconstruction
that asymmetry matters in a specific way: false positives are unprojected into
spurious 3D points and reduce cloud cleanliness, while missed boundary pixels
cannot be recovered downstream and reduce completeness.

## Iterative improvement

An earlier version of the dataset produced approximately 56% mAP@50. The final
dataset reaches 70.2%. Three changes were made to get there:

1. Reduced frame redundancy
2. Removal of stereo-duplicate contamination, that is, the same scene appearing
   via both the left and right camera
3. Addition of harder annotated examples

**This is not a controlled ablation.** All three changes were made together, so
the improvement shows that the earlier dataset construction was limiting
performance without establishing which change mattered most. The thesis states
this explicitly and it should not be presented as a single-factor result.

## Limitations

**No baseline detector was trained on the same split.** The result shows the
trained RF-DETR model is functional for this pipeline. It does not show that
RF-DETR is the right architecture for the task, because nothing else was tried
under comparable conditions.

**No training configuration survives.** Hyperparameters, augmentation settings,
epoch count and input resolution were all set in the Roboflow interface and are
not recoverable from this repository.

**Model version is ambiguous in the code.** The detection scripts referenced
Roboflow project versions 1, 2 and 3 across different files, and two class
naming schemes (`painted_line`/`track_edge` and the shorter `line`/`edge`).
`src/detect.py` defaults to version 2 and accepts both naming schemes, but which
version produced the figures above is not established.

## Two figures that disagree with the project notes

Flagged rather than silently resolved.

| Quantity | Thesis, Section 4.1 | Project notes |
|---|---|---|
| mAP, held-out test | 70.2% | 73% |
| Precision / recall | 82.6% / 69.6% (test) | 88.0% / 84.7% (validation) |

The precision and recall pair is probably not a conflict: the notes label those
figures as validation and the thesis reports test, which are different splits
and would be expected to differ in this direction.

The mAP figure is a real conflict. Both describe mAP on the held-out test set
and they differ by 2.8 points. The thesis is the submitted and graded artifact,
so its 70.2% is used throughout this repository. If 73% came from a later
retraining run, the thesis figure is the one that matches the reported precision,
recall and F1, and mixing the two would produce an internally inconsistent table.
