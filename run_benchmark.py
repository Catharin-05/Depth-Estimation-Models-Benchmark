"""
Run the full monocular-depth shootout: every model x every image in
IMAGES_DIR, recording latency + params, saving colorized depth maps, and
building a 3D point-cloud reconstruction + multi-view render for each.

Usage:
    python run_benchmark.py --images_dir ./images --out_dir ./outputs

Requires a machine with:
  - a real GPU (strongly recommended -- Marigold/Depth Pro/UniDepth/
    Metric3D are painfully slow on CPU)
  - unrestricted internet access to huggingface.co and github release
    assets (this script CANNOT run inside network-sandboxed environments
    that only allowlist pypi/github-code -- see README.md)

Each model in models/*.py exposes load_model() -> DepthModel with a
.predict(image) method (see models/base.py), so adding an 11th model is
just: write models/my_model.py, add one line to MODEL_REGISTRY below.
"""
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from common.utils import (
    count_params, benchmark_latency, estimate_intrinsics,
    depth_to_pointcloud, save_pointcloud_ply, colorize_depth,
    plot_pointcloud_views, save_json,
)

# Each entry: (registry key, module name, kwargs for load_model())
MODEL_REGISTRY = [
    ("midas_small", "models.midas", {"variant": "MiDaS_small"}),
    ("midas_dpt_hybrid", "models.midas", {"variant": "DPT_Hybrid"}),
    ("dpt_large", "models.dpt", {}),
    ("zoedepth_nk", "models.zoedepth", {"variant": "ZoeD_NK"}),
    ("depth_anything_v2_small", "models.depth_anything_v2", {"variant": "small"}),
    ("depth_anything_v2_large", "models.depth_anything_v2", {"variant": "large"}),
    ("depth_pro", "models.depth_pro", {}),
    ("unidepth_v2", "models.unidepth_v2", {"backbone": "vitl14"}),
    ("metric3d_v2", "models.metric3d", {"variant": "vit_large"}),
    ("marigold_lcm", "models.marigold", {}),
    ("patchfusion", "models.patchfusion", {}),
    ("leres", "models.leres", {}),
]


def run(images_dir: Path, out_dir: Path, hfov_deg: float = 60.0, only=None):
    import importlib

    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    results = {}

    for key, module_name, kwargs in MODEL_REGISTRY:
        if only and key not in only:
            continue

        print(f"\n=== {key} ===")
        try:
            module = importlib.import_module(module_name)
            t0 = time.time()
            wrapper = module.load_model(**kwargs)
            load_s = time.time() - t0
        except Exception as e:
            print(f"  [skip] failed to load: {e}")
            traceback.print_exc()
            results[key] = {"error": str(e)}
            continue

        model_dir = out_dir / key
        model_dir.mkdir(exist_ok=True)
        results[key] = {"name": wrapper.name, "params_M": wrapper.params_M,
                         "is_metric": wrapper.is_metric, "load_time_s": load_s,
                         "per_image": {}}

        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            rgb = np.array(image)

            try:
                lat = benchmark_latency(lambda: wrapper.predict(image), n_warmup=1, n_runs=3)
                depth, meta = wrapper.predict(image)
            except Exception as e:
                print(f"  [skip image {img_path.name}] {e}")
                traceback.print_exc()
                results[key]["per_image"][img_path.stem] = {"error": str(e)}
                continue

            # Save colorized depth map
            depth_vis = colorize_depth(depth)
            Image.fromarray(depth_vis).save(model_dir / f"{img_path.stem}_depth.png")
            np.save(model_dir / f"{img_path.stem}_depth.npy", depth)

            # Build + save 3D point cloud
            intr = estimate_intrinsics(rgb.shape[1], rgb.shape[0], hfov_deg=hfov_deg)
            depth_resized = np.array(Image.fromarray(depth).resize((rgb.shape[1], rgb.shape[0])))
            points, colors = depth_to_pointcloud(
                depth_resized, rgb, intr, is_relative=meta["is_relative"], stride=4
            )
            save_pointcloud_ply(str(model_dir / f"{img_path.stem}_cloud.ply"), points, colors)
            plot_pointcloud_views(
                points, colors, str(model_dir / f"{img_path.stem}_cloud_views.png"),
                title=f"{wrapper.name} -- {img_path.stem}",
            )

            results[key]["per_image"][img_path.stem] = {
                "latency_ms_mean": lat["mean_ms"],
                "latency_ms_std": lat["std_ms"],
                "n_points": int(points.shape[0]),
            }
            print(f"  {img_path.stem}: {lat['mean_ms']:.1f} ms/frame, {points.shape[0]} pts")

    save_json(results, str(out_dir / "results.json"))
    print(f"\nDone. Results written to {out_dir / 'results.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="./images")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--hfov_deg", type=float, default=60.0,
                         help="assumed horizontal FOV for point-cloud back-projection "
                              "when no real camera intrinsics are known")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                         help="subset of MODEL_REGISTRY keys to run, e.g. --only midas_small dpt_large")
    args = parser.parse_args()
    run(Path(args.images_dir), Path(args.out_dir), args.hfov_deg, args.only)
