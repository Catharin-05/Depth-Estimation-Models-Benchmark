"""
Quantitative depth benchmark on public datasets with real ground truth --
this is the pivot from the earlier no-ground-truth AI-image approach.

For each model x each dataset image:
  - runs inference, times it (same benchmark_latency() as before)
  - aligns scale/shift if the model is relative-depth (see eval/metrics.py)
  - computes AbsRel / RMSE / log10 / delta1-3 against real ground truth
  - builds a 3D point cloud using the dataset's REAL camera intrinsics
    (fixes the earlier reconstruction-quality problem entirely, since we're
    no longer guessing a 60deg FOV on images with unknown/impossible
    geometry)

Usage:
    python run_quant_benchmark.py --dataset nyu --n 20
    python run_quant_benchmark.py --dataset kitti --n 20
    python run_quant_benchmark.py --dataset both --n 20 --only midas_dpt_hybrid zoedepth_nk
"""
import argparse
import importlib
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from common.utils import (
    benchmark_latency, depth_to_pointcloud, save_pointcloud_ply,
    plot_pointcloud_views, colorize_depth, save_json,
)
from eval.datasets import load_nyu, load_kitti_eigen
from eval.metrics import evaluate_prediction, aggregate_metrics
from run_benchmark import MODEL_REGISTRY  # reuse the same model list


def run(dataset_name: str, n: int, out_dir: Path, only=None, save_clouds: bool = True,
        kitti_raw_dir: str = "./kitti_data/raw", kitti_depth_dir: str = "./kitti_data/depth_annotated"):
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    if dataset_name in ("nyu", "both"):
        samples += load_nyu(n=n)
    if dataset_name in ("kitti", "both"):
        samples += load_kitti_eigen(n=n, kitti_raw_dir=kitti_raw_dir, kitti_depth_dir=kitti_depth_dir)
    print(f"Loaded {len(samples)} samples ({dataset_name})")

    results = {}
    for key, module_name, kwargs in MODEL_REGISTRY:
        if only and key not in only:
            continue

        print(f"\n=== {key} ===")
        try:
            module = importlib.import_module(module_name)
            wrapper = module.load_model(**kwargs)
        except Exception as e:
            print(f"  [skip] failed to load: {e}")
            traceback.print_exc()
            results[key] = {"error": str(e)}
            continue

        model_dir = out_dir / key
        model_dir.mkdir(exist_ok=True)
        per_image_metrics = []
        latencies = []

        for sample in samples:
            try:
                lat = benchmark_latency(lambda: wrapper.predict(sample.rgb), n_warmup=1, n_runs=2)
                depth, meta = wrapper.predict(sample.rgb)
            except Exception as e:
                print(f"  [skip {sample.id}] inference failed: {e}")
                continue

            # Resize prediction to GT resolution if needed (models often
            # output at their own internal resolution)
            if depth.shape != sample.depth_gt.shape:
                depth = np.array(Image.fromarray(depth).resize(
                    (sample.depth_gt.shape[1], sample.depth_gt.shape[0]), Image.BILINEAR
                ))

            metrics = evaluate_prediction(
                depth, sample.depth_gt, is_metric=meta["is_metric"],
                max_depth=sample.max_depth, larger_is_closer=meta["larger_is_closer"],
            )
            metrics["latency_ms"] = lat["mean_ms"]
            metrics["sample_id"] = sample.id
            per_image_metrics.append(metrics)
            latencies.append(lat["mean_ms"])
            print(f"  {sample.id}: AbsRel={metrics.get('AbsRel', float('nan')):.3f} "
                  f"RMSE={metrics.get('RMSE', float('nan')):.3f} "
                  f"delta1={metrics.get('delta1', float('nan')):.3f} "
                  f"({lat['mean_ms']:.0f} ms)")

            if save_clouds:
                # Real intrinsics this time -- reconstructions should
                # actually look geometrically correct, not just plausible.
                rgb_arr = np.array(sample.rgb)
                points, colors = depth_to_pointcloud(
                    depth, rgb_arr, sample.intrinsics,
                    is_relative=not meta["is_metric"], stride=4, max_depth=sample.max_depth,
                )
                save_pointcloud_ply(str(model_dir / f"{sample.id}_cloud.ply"), points, colors)
                plot_pointcloud_views(
                    points, colors, str(model_dir / f"{sample.id}_cloud_views.png"),
                    title=f"{wrapper.name} -- {sample.id}",
                )
                Image.fromarray(colorize_depth(depth)).save(model_dir / f"{sample.id}_depth.png")

        results[key] = {
            "name": wrapper.name,
            "params_M": wrapper.params_M,
            "is_metric": wrapper.is_metric,
            "aggregate": aggregate_metrics(per_image_metrics),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
            "per_image": per_image_metrics,
        }

    save_json(results, str(out_dir / f"quant_results_{dataset_name}.json"))
    # Also write one file per model (harmless if run_all_isolated.py isn't
    # used) -- lets an isolated-subprocess run merge results afterward
    # without one process's results.json overwriting another's.
    for key, r in results.items():
        save_json(r, str(out_dir / f"partial_{dataset_name}_{key}.json"))
    print(f"\nDone. Results written to {out_dir / f'quant_results_{dataset_name}.json'}")

    # Quick leaderboard printout
    print(f"\n{'Model':<35}{'AbsRel':>8}{'RMSE':>8}{'delta1':>8}{'Latency(ms)':>13}")
    for key, r in results.items():
        if "aggregate" not in r or "error" in r.get("aggregate", {}):
            continue
        agg = r["aggregate"]
        print(f"{r['name']:<35}{agg['AbsRel']:>8.3f}{agg['RMSE']:>8.3f}{agg['delta1']:>8.3f}{r['mean_latency_ms']:>13.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nyu", "kitti", "both"], default="both")
    parser.add_argument("--n", type=int, default=20, help="samples per dataset")
    parser.add_argument("--out_dir", type=str, default="./outputs_quant")
    parser.add_argument("--only", type=str, nargs="*", default=None)
    parser.add_argument("--no_clouds", action="store_true", help="skip point cloud generation (faster)")
    parser.add_argument("--kitti_raw_dir", type=str, default="./kitti_data/raw")
    parser.add_argument("--kitti_depth_dir", type=str, default="./kitti_data/depth_annotated")
    args = parser.parse_args()
    run(args.dataset, args.n, Path(args.out_dir), args.only, save_clouds=not args.no_clouds,
        kitti_raw_dir=args.kitti_raw_dir, kitti_depth_dir=args.kitti_depth_dir)
