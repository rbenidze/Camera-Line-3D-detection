# Calibration limitation: Yas Marina metric scale

State this wherever Yas Marina results appear.

## What is missing

The racing team never supplied camera calibration files for the Yas Marina
recording. There are no intrinsics and no measured stereo baseline.

The values in `config/stereo_yas.json` are estimates, not measurements:

```json
{ "fx": 1280.0, "fy": 1280.0, "cx": 640.0, "cy": 360.0, "baseline": 0.32 }
```

The round numbers are the tell. `fx` and `fy` are set equal, the principal point
sits at exactly the image centre, and the baseline is a plausible guess rather
than a measurement. The filename was previously `stereo.json`, which was renamed
so a reader cannot mistake it for real calibration.

## What it does to the output

Depth comes from `Z = fx * B / d`, so an error in `fx` or `B` scales every
reconstructed distance by the same wrong factor. Concretely:

- Estimated rectification inflates disparity by roughly 10x.
- Yas Marina point clouds come out compressed 4 to 5 times in metric scale.
- The pit lane reads about 3 m across where it is about 12 m in reality.
- The true baseline is probably in the range 0.08 to 0.15 m, not the 0.32 m
  currently assumed.

Shape and relative geometry survive this. Absolute distances do not.

## Rectifying from estimated parameters did not fix it

The obvious response to missing calibration is to estimate it. That was tried,
across five iterations, ending in `experiments/rectify_yas.py`. It did not
produce usable metric output, and the script ships as a record of the attempt
rather than as a working stage of the pipeline. Nothing downstream depends on
it.

The reported estimate was computed from 5 frames drawn from within the first 40
of the recording, so it was never sampled across the run. That is a real
weakness in the estimate, but it is not the reason the approach failed. Raising
the sample count would tighten the fit and leave the scale error untouched.

The approach was a three-axis corrective search over pitch, roll and yaw,
minimising the residual vertical offset between matched features across several
frames. It can drive that offset down. That is not the same as recovering the
geometry, because nothing in the objective constrains horizontal scale: rows can
be made to line up while the disparity they produce remains wrong by a constant
factor. Which is what happened. Disparity from a pair rectified this way is
inflated by roughly 10x, and the point clouds built from it are compressed 4 to
5 times.

The reason is not that the search was too coarse or evaluated on too few frames.
It is that a scale which was never measured cannot be recovered from the images
alone. Rectification fixes the relationship between two views. It does not
supply the baseline.

## Three different focal lengths for the same camera

This is the clearest evidence in the repository that the estimated intrinsics
are unreliable. Three values for `fx` are in active use, all describing the same
1280x720 front-left camera:

| `fx` | Where | Used for |
|---|---|---|
| 1280.0 | `config/stereo_yas.json`, field `fx` | read by `src/unproject.py` |
| 640.0 | `tools/build_lidar_gt.py`, `tools/project_lidar.py` | placing LiDAR points on pixels |
| 369.5 | `experiments/rectify_yas.py`, implied by the assumed fov of 120 degrees | the rectification search |

A 3.5x spread. None of the three is measured. Each was chosen to make a
particular step behave, which is what happens when the real value is unavailable.

The 640.0 combination is the one whose LiDAR overlay was actually validated
against real geometry, so it is the only one with any empirical support, and
that support is qualitative rather than metric.

They are deliberately not merged. `config/stereo_yas.json` carries `fx` and a
separate `fx_projection` precisely so that reading one does not silently change
the other. Collapsing them into a single `fx` would move every projected LiDAR
point by a factor of two and break the one transform that was validated. The
duplication is a symptom, not the disease, and the fix is a calibration file
rather than a tidier config.

## What it is not

This is a data availability problem, not a pipeline correctness problem. The
same code produces metrically valid output on DrivingStereo, which ships real
calibration (`config/stereo_drivingstereo.json`, `fx = 1003.556`,
`B = 0.5446`, derived from `P_rect_103`). Every metrically valid figure reported
in the thesis comes from DrivingStereo for exactly this reason.

## Evidence trail

The investigation that established this ran through a chain of scripts kept in
`_archive/`. They are not shipped because each one hardcodes paths to the
recording and frame directories, so they are not runnable by anyone else. The
findings they produced:

| Finding | Source |
|---|---|
| The front-left camera is mounted with a roll and yaw offset that every earlier script ignored by assuming an identity rotation. Fixing it via the URDF is what finally made LiDAR land on real geometry. | `yas_overlay_fixed.py` |
| The correct rotation convention had to be found by sweeping plausible compose orders, then corrected with an added downward pitch. | `yas_overlay_sweep.py`, `yas_pitch.py` |
| The YAML principal point is not the image centre. Using the image centre put projected points above the road. | `test_intrinsics2.py` |
| The left and right camera topics are not index synchronised, producing a per-frame varying horizontal offset (78 px on one frame, 300 px on another) rather than a fixed rectification constant. | `fl_fr_sync.py` |
| Naive plane fitting for scale recovery fails in this scene because RANSAC keeps selecting building walls instead of the road. Restricting both clouds to a narrow front-centre corridor fixes it. | `tools/recover_scale.py` |
| Focal length can be recovered against the LiDAR flat-road reference. | `tools/recover_fx.py` |

Index-based frame to sweep pairing was also shown to be wrong: it produced a
five-fold spread in recovered `fx * B` (2292, 1711, 8425, 3528) instead of a
tight cluster. Pairing by timestamp is what `tools/add_timestamps.py` exists to
enable.
