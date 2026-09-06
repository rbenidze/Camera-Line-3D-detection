# RAFT-Stereo fine-tuning experiments

Source for every number here: [raft_finetune_colab.ipynb](raft_finetune_colab.ipynb),
whose stored cell outputs are preserved. Image outputs were stripped to cut the
file from 6.97 MB to 0.16 MB; all 78 text outputs, which are the actual results,
were kept.

## The finding

All three fine-tuning runs beat the pretrained Scene Flow baseline on aggregate
disparity error, and all three got **worse** on the thin structures this project
exists to reconstruct. The deployed model is therefore `raftstereo-sceneflow.pth`,
the pretrained weights. The fine-tuned checkpoints are kept as evidence, not
deployed.

The cause is supervision quality, not dataset size and not hyperparameters.
LiDAR-derived ground truth is sparse and biased toward large planar surfaces, so
training against it teaches the network to over-smooth exactly the painted lines
and curb edges the pipeline needs. Section 2 is the measurement that shows this,
rather than an assertion about it.

## 1. Disparity results

From cell 8 of [the notebook](raft_finetune_colab.ipynb), evaluated on its
21-frame DrivingStereo holdout. These are the reported numbers.

| Model | EPE (px) | bad-3 (%) |
|---|---|---|
| Pretrained Scene Flow baseline | 0.859 | 2.74 |
| Fine-tuned, DrivingStereo only | 0.515 | 0.61 |
| Fine-tuned, DrivingStereo + KITTI | 0.547 | 0.76 |

Per-checkpoint progression for the DrivingStereo run, same cell:

| Step | EPE (px) | bad-3 (%) |
|---|---|---|
| 500 | 0.565 | 0.89 |
| 1000 | 0.536 | 0.68 |
| 1500 | 0.526 | 0.69 |
| **2000** | **0.515** | **0.61** |
| 2500 | 0.517 | 0.65 |

Step 2000 was selected as the best checkpoint. Training longer made it worse,
which is why checkpoint selection was by held-out metric rather than by final
step.

Read on its own this table says fine-tuning worked. Section 2 is why that
reading is wrong.

## 2. Over-smoothing: the mechanism, measured

This is the core finding, and it is a measurement, not an inference.

Cell 26 of [the notebook](raft_finetune_colab.ipynb) splits the error by pixel
type, separating high-gradient (edge) pixels from flat regions:

| Model | edge EPE | flat EPE | gap |
|---|---|---|---|
| Fine-tuned, DrivingStereo only | 0.561 | 0.519 | 0.042 |
| Fine-tuned, DrivingStereo + KITTI | 0.631 | 0.539 | 0.092 |

Both models are worse on edges than on flat surfaces, which is expected. What
matters is the direction of change: **adding more sparse LiDAR-supervised data
more than doubled the gap**, from 0.042 to 0.092. Edge EPE rose from 0.561 to
0.631 while flat EPE barely moved, 0.519 to 0.539. The extra data did not
degrade the model uniformly. It degraded it specifically on thin structures.

That is the mechanism. Sparse LiDAR returns land overwhelmingly on large planar
surfaces, so a network trained against them is rewarded for smoothing and is
never penalised for erasing a painted line, because there is rarely a ground
truth point on the line to disagree with. More of that supervision means more
smoothing pressure.

Stated plainly: every fine-tune beat the baseline on aggregate EPE while getting
worse on exactly the thin structures this project needs. Aggregate EPE is
dominated by flat road and wall pixels, so it rewards the very behaviour that
breaks the task. That is why `raftstereo-sceneflow.pth` is deployed and the
fine-tuned checkpoints are not.

The 3D consequence appears in section 4: 39.5 cm error on high-gradient pixels
against 21.9 cm over all pixels.

## 3. Training commands and holdout method

Both runs, verbatim from cells 7 and 17 of
[the notebook](raft_finetune_colab.ipynb). Both start from the pretrained Scene
Flow weights, both ran on a Colab T4, and both use the patched
`train_stereo.py` and `core/stereo_datasets.py` from `patches/`.

**Run 1, DrivingStereo only:**

```bash
python train_stereo.py --name ds_finetune_v2 \
    --restore_ckpt models/raftstereo-sceneflow.pth \
    --train_datasets drivingstereo \
    --num_steps 2500 --batch_size 4 \
    --train_iters 16 --valid_iters 32 \
    --image_size 288 720 --mixed_precision --lr 0.00001
```

**Run 2, DrivingStereo + KITTI:**

```bash
python train_stereo.py --name ds_kitti_finetune_v1 \
  --restore_ckpt models/raftstereo-sceneflow.pth \
  --train_datasets drivingstereo kitti \
  --num_steps 2500 --batch_size 4 \
  --train_iters 16 --valid_iters 32 \
  --image_size 288 720 --mixed_precision --lr 0.00001
```

Identical hyperparameters. The only variable is the training set, which is what
makes the widening edge gap in section 2 attributable to the added data.

The `--image_size 288 720` follows the sky crop applied by
`tools/crop_dataset.py`, which reduces 881x400 to 881x320.

**Holdout method in the notebook.** Cell 8 built its evaluation set as:

```python
holds = sorted(glob.glob('datasets/DrivingStereo/left/*/*.jpg'))[::400]
```

A flat stride of 400 across every sequence globbed together, giving 21 frames.
Sampling sparsely is deliberate: consecutive DrivingStereo frames are near
duplicates, and near-identical frames on both sides of a split make the
evaluation overstate accuracy.

## 4. 3D reconstruction accuracy

From cell 38 of [the notebook](raft_finetune_colab.ipynb):

| Quantity | Value |
|---|---|
| Median 3D error, all pixels | 21.9 cm |
| Median 3D error, high-gradient pixels | 39.5 cm |
| Median depth-only error (Z) | 21.2 cm |
| Median GT holdout depth | 16.2 m (range 12.3 to 22.4 m) |

Three caveats attach to these figures.

**The checkpoint.** Cell 38 loads `checkpoints/2000_ds_finetune_v2.pth`, the
Run 1 step-2000 fine-tuned model, not the deployed Scene Flow weights. So 21.9 cm
characterises the fine-tuned checkpoint on the DrivingStereo holdout while the
deployed model is `raftstereo-sceneflow.pth`. Both statements are true and they
are easy to conflate.

**The aggregation.** Per frame the code takes a median over valid pixels, then
averages those per-frame medians with `np.mean` across frames. It is reported as
a median. `src/evaluate.py` preserves this arithmetic exactly, because changing
it would change a locked number.

**The split.** Cell 38 uses its own frame selection, distinct from cell 8's. See
section 5.

"High-gradient pixels" means a Sobel gradient magnitude above its 90th
percentile. It is not the RF-DETR `painted_line` or `track_edge` mask. The thesis
calls this metric "boundary/edge pixels", which overstates what is measured: it
is a proxy for thin-structure error, computed on image gradient rather than on
detected track boundaries.

## 5. Known discrepancies

[The notebook](raft_finetune_colab.ipynb) disagrees with itself about the
holdout, and the disagreement moves the numbers.

| Source | Frames | EPE (px) | bad-3 (%) |
|---|---|---|---|
| Cell 8 (reported) | 21 | 0.515 | 0.61 |
| Cells 26, 31, 32 | 22 | 0.520 | 0.75 |

Notebook (3) also labels its chart "21 held-out DrivingStereo frames".

The cause is exact, not mysterious. Cell 8 strides across all sequences globbed
together:

```python
sorted(glob.glob('datasets/DrivingStereo/left/*/*.jpg'))[::400]
```

Cells 26, 31 and 32 stride **within each sequence** instead:

```python
for seq in sorted(os.listdir(root/'left')):
    ls = sorted(glob.glob(root/'left'/seq/'*.jpg'))
    for i in range(0, len(ls), 400):
```

A per-sequence stride restarts the count at every sequence boundary, so short
sequences contribute a frame they would not contribute under a flat stride. Here
that adds exactly one frame, and that single frame moves bad-3 by 0.14
percentage points, roughly 23% of its own value.

**Cell 8's numbers are the reported ones**, and cell 8's flat stride is the
method `tools/make_holdout.py` implements.

This is precisely the class of drift the manifest exists to prevent. Both cells
believed they were computing "the holdout" and neither recorded which frames it
contained, so the difference was invisible until someone compared outputs.
`tools/make_holdout.py` writes `config/holdout_manifest.json` naming every
held-out frame, and `src/evaluate.py` reads that file rather than re-deriving a
stride, so the evaluation set is a recorded fact instead of a recomputation over
whatever happens to be on disk.

## 6. Reproduction status

**`src/evaluate.py` will hard-fail until you create the holdout.** This is by
design, and it is worth stating rather than leaving someone to discover it.

`config/holdout_manifest.json` does not exist in this repository. It cannot be
committed, because it names frames in a DrivingStereo tree that is not
distributed here. Until it is generated, `src/evaluate.py` exits with a non-zero
status and a message telling you to run `tools/make_holdout.py` first. It will
not fall back to evaluating on the full dataset, because that would silently
score the model on frames it may have trained on.

To reproduce:

```bash
python tools/make_holdout.py --root datasets/DrivingStereo \
    --holdout datasets/DrivingStereo_holdout --every 400

python src/evaluate.py --holdout-root datasets/DrivingStereo_holdout \
    --manifest config/holdout_manifest.json \
    --checkpoint checkpoints/2000_ds_finetune_v2.pth \
    --calib config/stereo_drivingstereo.json
```

**Expect the numbers to differ from section 4.** The reported figures came from
the notebook's own re-derived stride over whatever was on disk in June 2026. The
manifest path is the reproducible one; the notebook figures are the historical
record. Neither should be edited to match the other. If they differ materially,
that is a finding about the split, not an error to paper over, and section 5 is
the reason to expect it.

See also [calibration-limitation.md](calibration-limitation.md) for why none of
this transfers to Yas Marina metric output.
