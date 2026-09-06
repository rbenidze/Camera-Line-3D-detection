"""
Stage 2: dense disparity with RAFT-Stereo.

Extracted from the Colab notebook (raft_finetune_colab_v4.ipynb) and from
calibofdrivingstereo.py, both of which hardcoded /content/RAFT-Stereo. The
upstream repo location is a parameter here instead.

DEPLOYED MODEL: raftstereo-sceneflow.pth, the pretrained Scene Flow weights.
All three fine-tuning runs (KITTI x2, DrivingStereo x1) produced worse disparity
on the target domain than these pretrained weights, so the fine-tuned
checkpoints are kept as evidence and are not deployed. See docs/experiment-log.md.

Requires the upstream RAFT-Stereo repo, with patches/ applied:
    git clone https://github.com/princeton-vl/RAFT-Stereo
    git apply /path/to/patches/*.patch

Usage:
    # one pair
    python src/disparity.py --left l.png --right r.png \
        --checkpoint models/raftstereo-sceneflow.pth --output disp.npy

    # a directory of matched pairs
    python src/disparity.py --left-dir left/ --right-dir right/ \
        --checkpoint models/raftstereo-sceneflow.pth --output-dir disp/
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

# Architecture of the released RAFT-Stereo checkpoints. These must match the
# weights being loaded; they are not tuning knobs.
MODEL_ARGS = Namespace(
    hidden_dims=[128] * 3,
    corr_implementation="reg",
    shared_backbone=False,
    corr_levels=4,
    corr_radius=4,
    n_downsample=2,
    context_norm="batch",
    slow_fast_gru=False,
    n_gru_layers=3,
    mixed_precision=True,
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def add_raft_stereo_to_path(repo: Path):
    """Put the upstream RAFT-Stereo repo on sys.path so `core` imports resolve."""
    if not repo.exists():
        sys.exit(
            f"RAFT-Stereo repo not found at {repo}.\n"
            "Pass --raft-stereo or set the RAFT_STEREO_PATH environment variable."
        )
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "core"))


def load_model(checkpoint: Path):
    """Load RAFT-Stereo weights onto the GPU in eval mode."""
    import torch
    from raft_stereo import RAFTStereo

    model = torch.nn.DataParallel(RAFTStereo(MODEL_ARGS), device_ids=[0])
    model.load_state_dict(torch.load(str(checkpoint)))
    return model.module.cuda().eval()


def read_image(path: Path):
    import torch

    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].cuda()


def infer_pair(model, left: Path, right: Path, iters: int) -> np.ndarray:
    """Predict disparity for one rectified stereo pair, in pixels."""
    import torch
    from utils.utils import InputPadder

    with torch.no_grad():
        a, b = read_image(left), read_image(right)
        padder = InputPadder(a.shape, divis_by=32)
        a, b = padder.pad(a, b)
        _, flow_up = model(a, b, iters=iters, test_mode=True)
        return np.abs(padder.unpad(flow_up).cpu().numpy().squeeze())


def save_disparity(disp: np.ndarray, out_path: Path, preview: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path.with_suffix(".npy"), disp)
    print(f"[save]  -> {out_path.with_suffix('.npy')}  "
          f"median={np.median(disp):.2f} px")
    if preview:
        import matplotlib.pyplot as plt
        plt.imsave(str(out_path.with_suffix(".png")), -disp, cmap="jet")
        print(f"[save]  -> {out_path.with_suffix('.png')}")


def matched_pairs(left_dir: Path, right_dir: Path):
    """Pair left and right images by filename."""
    pairs = []
    for path in sorted(left_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        right = right_dir / path.name
        if right.exists():
            pairs.append((path, right))
    if not pairs:
        sys.exit(f"No matching pairs between {left_dir} and {right_dir}")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="RAFT-Stereo disparity inference.")
    parser.add_argument("--left", type=Path, help="left image of a single pair")
    parser.add_argument("--right", type=Path, help="right image of a single pair")
    parser.add_argument("--left-dir", type=Path, help="directory of left images")
    parser.add_argument("--right-dir", type=Path, help="directory of right images")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="model weights, e.g. raftstereo-sceneflow.pth")
    parser.add_argument("--output", type=Path,
                        help="output path for a single pair (.npy)")
    parser.add_argument("--output-dir", type=Path,
                        help="output directory for a batch")
    parser.add_argument("--raft-stereo", type=Path,
                        default=Path(os.environ.get("RAFT_STEREO_PATH",
                                                    Path.home() / "RAFT-Stereo")),
                        help="path to the upstream RAFT-Stereo clone")
    parser.add_argument("--iters", type=int, default=32,
                        help="refinement iterations (default 32; the notebook "
                             "evaluation used 16)")
    parser.add_argument("--preview", action="store_true",
                        help="also write a colourised .png next to each .npy")
    args = parser.parse_args()

    single = args.left and args.right and args.output
    batch = args.left_dir and args.right_dir and args.output_dir
    if not single and not batch:
        parser.error("pass either --left/--right/--output or "
                     "--left-dir/--right-dir/--output-dir")

    add_raft_stereo_to_path(args.raft_stereo)
    print(f"[model] loading {args.checkpoint}")
    model = load_model(args.checkpoint)

    if single:
        disp = infer_pair(model, args.left, args.right, args.iters)
        save_disparity(disp, args.output, args.preview)
        return

    pairs = matched_pairs(args.left_dir, args.right_dir)
    print(f"[batch] {len(pairs)} pairs")
    for idx, (left, right) in enumerate(pairs, 1):
        disp = infer_pair(model, left, right, args.iters)
        print(f"[{idx}/{len(pairs)}] {left.name}")
        save_disparity(disp, args.output_dir / left.stem, args.preview)


if __name__ == "__main__":
    main()
