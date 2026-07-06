# Monocular Depth Estimation

Benchmarks 10 depth models (MiDaS, DPT, ZoeDepth, Depth Anything V2, Depth Pro,
UniDepth V2, Metric3D v2, Marigold, PatchFusion, LeReS) on your own images:
latency, parameter counts, depth maps, and 3D point-cloud reconstructions.

## Setup

```bash
pip install -r requirements.txt
```

Some models need extra manual setup (see docstrings at the top of each
`models/*.py` file for specifics):

- **UniDepth V2, Metric3D v2, PatchFusion** — pulled live via `torch.hub` from
  their GitHub repos on first run. If a repo's own dependencies aren't
  installed, `git clone` it and `pip install -r requirements.txt` from there.
- **LeReS** — no pip/hub packaging at all. Clone
  `https://github.com/aim-uofa/AdelaiDepth` and download `res101.pth` per
  their README, then point `models/leres.py`'s `load_model()` at the paths.
- **Marigold** — needs `diffusers` + `accelerate`; GPU strongly recommended
  (diffusion sampling, even distilled to 4 steps, is much slower than a
  single forward pass).
- A real **GPU** is recommended overall — Depth Pro, UniDepth V2, Metric3D,
  and Marigold are heavy on CPU.

## Run

### Option A — Quantitative benchmark on public datasets (recommended)

Evaluates against real ground truth (NYU Depth V2 indoor, KITTI Eigen split
outdoor), with real camera intrinsics so 3D reconstructions are actually
geometrically correct, not just plausible-looking.

**NYU Depth V2** loads automatically, no account needed:
```bash
python run_quant_benchmark.py --dataset nyu --n 20
```
(Uses `jagennath-hari/nyuv2`, a script-free/parquet HF mirror -- the
originally-tried repos, `sayakpaul/nyu_depth_v2` and `0jl/NYUv2`, both rely
on a dataset loading *script*, which `datasets>=4.0` no longer supports for
security reasons -- that's the exact error you hit.)

**KITTI Eigen split** requires a one-time manual download (free
registration, no reliable script-free HF mirror exists for this one):
1. Register at https://www.cvlibs.net/datasets/kitti/user_register.php
2. Download the raw sync'd drives from
   https://www.cvlibs.net/datasets/kitti/raw_data.php (see
   `EIGEN_TEST_FILES` in `eval/datasets.py` for exactly which drives you need)
3. Download `data_depth_annotated.zip` from
   https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
4. Point the CLI at your extracted folders:
```bash
python run_quant_benchmark.py --dataset kitti --n 20 \
    --kitti_raw_dir /path/to/kitti_raw --kitti_depth_dir /path/to/kitti_depth_annotated
```

Or run both together (`--dataset both`) once KITTI is set up, or subset
models with `--only midas_dpt_hybrid zoedepth_nk`.

Outputs per model in `outputs_quant/<model_key>/`: colorized depth maps,
`.ply` point clouds + view renders (real intrinsics this time), plus
`outputs_quant/quant_results_<dataset>.json` with AbsRel/RMSE/log10/delta1-3
per image and averaged per model, and a quick leaderboard printed to stdout.

Relative-depth models (MiDaS, DPT, Depth Anything V2, Marigold, LeReS) are
automatically scale/shift-aligned to ground truth before scoring -- see
`eval/metrics.py` docstring for why that's the correct (and standard)
comparison protocol, not a shortcut.

### Option B — Qualitative run on your own images (no ground truth)

```bash
# put your images (.png/.jpg) in ./images first
python run_benchmark.py --images_dir ./images --out_dir ./outputs
```

## What you get

For each model, in `outputs/<model_key>/`:
- `<image>_depth.png` — colorized depth map
- `<image>_depth.npy` — raw depth array
- `<image>_cloud.ply` — 3D point cloud (open in MeshLab / CloudCompare / any .ply viewer)
- `<image>_cloud_views.png` — front/side/top-down render of the point cloud

Plus `outputs/results.json` with params, load time, and per-image latency
(mean/std/min/max) for every model.

## Files

- `common/utils.py` — latency benchmarking, param counting, depth→point-cloud
  back-projection (pinhole model), point cloud / depth-map visualization
- `models/base.py` — shared `DepthModel` interface every wrapper implements
- `models/*.py` — one file per model, following each project's officially
  documented usage (`transformers` pipeline, `torch.hub`, or `diffusers`
  depending on the model)
- `run_benchmark.py` — orchestrates everything above across all images/models

## Notes on 3D reconstruction

`common/utils.estimate_intrinsics()` assumes a 60° horizontal FOV since most
images won't have real camera intrinsics (EXIF focal length, etc.). If you
know the real intrinsics, pass them directly to `depth_to_pointcloud()`
instead — reconstructions will be metrically correct rather than just
plausible-looking.

Relative-depth models (MiDaS, DPT, Depth Anything V2, Marigold, LeReS) only
give depth up to an unknown scale + shift — their `is_metric=False` flag
handles this in the reconstruction code, but treat their "3D scenes" as
qualitative, not quantitative.

## Robustness: isolated per-model runs (macOS Fatal Python error)

Running many heavy models back-to-back in one process can occasionally hit
a `Fatal Python error` on macOS -- not a catchable Python exception, related
to Metal/MPS state plus subprocess forking (torch.hub git operations,
huggingface_hub downloads) not playing well together once MPS has been
touched in that process. If you hit that, use the isolated runner instead:

```bash
python run_all_isolated.py --dataset nyu --n 20
```

It runs each model in its own fresh Python process, so one crashing doesn't
take the rest of the benchmark down with it. Writes
`outputs_quant/quant_results_<dataset>_merged.json` and reports which model
keys (if any) crashed, so you can retry just those with `--only`.
