"""
Load small evaluation subsets of NYU Depth V2 (indoor) and KITTI Eigen
split (outdoor) -- the two standard zero-shot depth benchmarks that every
paper in this space (ZoeDepth, Depth Anything V2, UniDepth V2, Metric3D
v2, Depth Pro...) reports numbers on. Using these means your results are
directly comparable to published leaderboards, and -- unlike the
AI-generated images -- both ship REAL camera intrinsics, so 3D
reconstructions will be geometrically correct instead of "plausible".

Usage:
    from eval.datasets import load_nyu, load_kitti_eigen
    samples = load_nyu(n=20)      # -> list[Sample]
    samples = load_kitti_eigen(n=20)

Each Sample has: rgb (PIL.Image), depth_gt (HxW meters, float32),
intrinsics (dict fx,fy,cx,cy), dataset, id.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class Sample:
    id: str
    dataset: str
    rgb: Image.Image
    depth_gt: np.ndarray  # meters, 0 = invalid/no groundtruth
    intrinsics: dict       # fx, fy, cx, cy
    max_depth: float        # dataset-standard evaluation cap (NYU=10m, KITTI=80m)


# --------------------------------------------------------------------------
# NYU Depth V2 (indoor) -- official capture intrinsics, from the dataset's
# camera_params.m / toolbox (Silberman et al., 2012). These are the commonly
# cited default intrinsics for the raw Kinect RGB camera.
# --------------------------------------------------------------------------
NYU_INTRINSICS = {"fx": 518.857901, "fy": 519.469611, "cx": 325.582449, "cy": 253.736166}
NYU_MAX_DEPTH = 10.0


def load_nyu(n: int = 20, split: str = "test") -> list[Sample]:
    """
    Uses jagennath-hari/nyuv2 -- confirmed script-free/parquet-native on HF
    (the earlier candidates, sayakpaul/nyu_depth_v2 and 0jl/NYUv2, both
    rely on a dataset loading *script*, which datasets>=4.0 no longer
    supports -- that's the exact error you hit). This repo stores depth as
    uint16 PNG/TIFF in millimeters with a separate scaling_factors.json,
    so we download that once and convert to float32 meters.
    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    import json as _json
    import os as _os

    repo_id = "jagennath-hari/nyuv2"
    try:
        ds = load_dataset(repo_id, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Could not load {repo_id} (split={split}). Error: {e}\n"
            "Check https://huggingface.co/datasets/jagennath-hari/nyuv2 for "
            "current split names, or fall back to the original NYU release: "
            "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html "
            "(nyu_depth_v2_labeled.mat, readable with h5py)."
        )

    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns="scaling_factors.json")
    with open(_os.path.join(local_dir, "scaling_factors.json")) as f:
        scale = _json.load(f)
    depth_scale = scale["depth_scale"]  # divide raw uint16 by this to get meters

    samples = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        rgb = row["rgb"].convert("RGB")
        depth_mm = np.array(row["depth"], dtype=np.float32)
        depth_m = depth_mm / depth_scale
        samples.append(Sample(
            id=f"nyu_{i:03d}", dataset="nyu", rgb=rgb, depth_gt=depth_m,
            intrinsics=NYU_INTRINSICS, max_depth=NYU_MAX_DEPTH,
        ))
    return samples


# --------------------------------------------------------------------------
# KITTI Eigen split (outdoor driving). Unlike NYU, I could NOT find a
# reliable script-free HF mirror for this during testing -- rather than
# guess another repo name (which is exactly what broke NYU last time),
# this uses the real, standard path every paper in this space actually
# uses: raw KITTI + the official Eigen test split file list.
#
# Setup (one-time, ~free but requires registering with KITTI):
#   1. Register at https://www.cvlibs.net/datasets/kitti/user_register.php
#   2. Download the raw sync'd drives listed in EIGEN_TEST_DRIVES below from
#      https://www.cvlibs.net/datasets/kitti/raw_data.php
#      (you only need the ~29 drives that appear in the test split, not all
#      of KITTI raw -- see the drive names embedded below)
#   3. Download "annotated depth maps" (data_depth_annotated.zip) from
#      https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
#   4. Point --kitti_raw_dir / --kitti_depth_dir at the extracted folders:
#        kitti_raw_dir/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/*.png
#        kitti_depth_dir/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/*.png
#
# EIGEN_TEST_FILES below is a real subset (first ~40 of 697) of the
# official split, verified against
# github.com/nianticlabs/monodepth2/blob/master/splits/eigen/test_files.txt
# -- format: "<date>/<drive> <frame_idx> <side>" (side 'l' = left/image_02).
# --------------------------------------------------------------------------
EIGEN_TEST_FILES = """
2011_09_26/2011_09_26_drive_0002_sync 0000000069 l
2011_09_26/2011_09_26_drive_0002_sync 0000000054 l
2011_09_26/2011_09_26_drive_0002_sync 0000000042 l
2011_09_26/2011_09_26_drive_0002_sync 0000000057 l
2011_09_26/2011_09_26_drive_0002_sync 0000000030 l
2011_09_26/2011_09_26_drive_0002_sync 0000000027 l
2011_09_26/2011_09_26_drive_0002_sync 0000000012 l
2011_09_26/2011_09_26_drive_0002_sync 0000000075 l
2011_09_26/2011_09_26_drive_0002_sync 0000000036 l
2011_09_26/2011_09_26_drive_0002_sync 0000000033 l
2011_09_26/2011_09_26_drive_0002_sync 0000000015 l
2011_09_26/2011_09_26_drive_0002_sync 0000000072 l
2011_09_26/2011_09_26_drive_0002_sync 0000000003 l
2011_09_26/2011_09_26_drive_0002_sync 0000000039 l
2011_09_26/2011_09_26_drive_0002_sync 0000000009 l
2011_09_26/2011_09_26_drive_0002_sync 0000000051 l
2011_09_26/2011_09_26_drive_0002_sync 0000000060 l
2011_09_26/2011_09_26_drive_0002_sync 0000000021 l
2011_09_26/2011_09_26_drive_0002_sync 0000000000 l
2011_09_26/2011_09_26_drive_0002_sync 0000000024 l
2011_09_26/2011_09_26_drive_0009_sync 0000000000 l
2011_09_26/2011_09_26_drive_0009_sync 0000000016 l
2011_09_26/2011_09_26_drive_0009_sync 0000000032 l
2011_09_26/2011_09_26_drive_0009_sync 0000000048 l
2011_09_26/2011_09_26_drive_0009_sync 0000000064 l
2011_09_26/2011_09_26_drive_0013_sync 0000000090 l
2011_09_26/2011_09_26_drive_0013_sync 0000000050 l
2011_09_26/2011_09_26_drive_0013_sync 0000000110 l
2011_09_26/2011_09_26_drive_0013_sync 0000000115 l
2011_09_26/2011_09_26_drive_0013_sync 0000000060 l
2011_09_26/2011_09_26_drive_0020_sync 0000000003 l
2011_09_26/2011_09_26_drive_0020_sync 0000000069 l
2011_09_26/2011_09_26_drive_0020_sync 0000000057 l
2011_09_26/2011_09_26_drive_0020_sync 0000000012 l
2011_09_26/2011_09_26_drive_0020_sync 0000000072 l
2011_09_26/2011_09_26_drive_0023_sync 0000000018 l
2011_09_26/2011_09_26_drive_0023_sync 0000000090 l
2011_09_26/2011_09_26_drive_0023_sync 0000000126 l
2011_09_26/2011_09_26_drive_0023_sync 0000000378 l
2011_09_26/2011_09_26_drive_0023_sync 0000000036 l
""".strip().splitlines()

KITTI_INTRINSICS = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.854}
KITTI_MAX_DEPTH = 80.0


def load_kitti_eigen(n: int = 20, kitti_raw_dir: str = "./kitti_data/raw",
                      kitti_depth_dir: str = "./kitti_data/depth_annotated") -> list[Sample]:
    """
    Loads n samples from the real KITTI Eigen test split, reading local
    files -- see the setup steps in the module docstring above. This does
    NOT auto-download (KITTI requires registration), but does the correct
    thing once you've downloaded it, instead of guessing at a possibly
    nonexistent/renamed HF mirror.
    """
    import os

    if not os.path.isdir(kitti_raw_dir):
        raise RuntimeError(
            f"'{kitti_raw_dir}' not found. KITTI requires manual download "
            "(free registration) -- see the setup steps in eval/datasets.py's "
            "module docstring above load_kitti_eigen(), or pass --dataset nyu "
            "to run NYU only for now."
        )

    samples = []
    for i, line in enumerate(EIGEN_TEST_FILES):
        if i >= n:
            break
        drive_path, frame_idx, side = line.split()
        cam = "image_02" if side == "l" else "image_03"
        drive_name = drive_path.split("/")[1]

        rgb_path = os.path.join(kitti_raw_dir, drive_path, cam, "data", f"{frame_idx}.png")
        depth_path = os.path.join(kitti_depth_dir, drive_name, "proj_depth", "groundtruth", cam, f"{frame_idx}.png")

        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            print(f"  [skip] missing local files for {drive_path} {frame_idx} (expected at {rgb_path})")
            continue

        rgb = Image.open(rgb_path).convert("RGB")
        # KITTI's official annotated depth convention: 16-bit PNG, depth_m = png_value / 256.0, 0 = invalid
        depth_png = np.array(Image.open(depth_path), dtype=np.float32)
        depth_m = depth_png / 256.0

        samples.append(Sample(
            id=f"kitti_{i:03d}", dataset="kitti", rgb=rgb, depth_gt=depth_m,
            intrinsics=KITTI_INTRINSICS, max_depth=KITTI_MAX_DEPTH,
        ))
    return samples
