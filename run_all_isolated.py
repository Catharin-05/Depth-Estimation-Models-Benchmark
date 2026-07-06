"""
Runs run_quant_benchmark.py once per model, each in its OWN fresh Python
process, then merges the results.

Why this exists: a "Fatal Python error" (as opposed to a normal Python
exception) bypasses every try/except in run_quant_benchmark.py entirely --
it can't be caught, only avoided. On macOS this shows up when a process
has touched Metal/MPS and then something later in that same process forks
or spawns a subprocess (torch.hub's git operations, huggingface_hub
downloads, etc. can trigger this) -- Apple's Objective-C runtime isn't
fork-safe once Metal has been initialized. Running each model in its own
process sidesteps the whole problem: if one model's process crashes hard,
you keep every other model's results instead of losing the whole run.

Usage:
    python run_all_isolated.py --dataset nyu --n 20
    python run_all_isolated.py --dataset both --n 20 --only midas_dpt_hybrid zoedepth_nk depth_pro
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path

from run_benchmark import MODEL_REGISTRY
from common.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nyu", "kitti", "both"], default="nyu")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="./outputs_quant")
    parser.add_argument("--only", type=str, nargs="*", default=None)
    parser.add_argument("--kitti_raw_dir", type=str, default="./kitti_data/raw")
    parser.add_argument("--kitti_depth_dir", type=str, default="./kitti_data/depth_annotated")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = args.only or [k for k, _, _ in MODEL_REGISTRY]

    crashed = []
    for key in keys:
        print(f"\n{'='*60}\nRunning {key} in its own process...\n{'='*60}")
        cmd = [
            sys.executable, "run_quant_benchmark.py",
            "--dataset", args.dataset, "--n", str(args.n),
            "--out_dir", str(out_dir), "--only", key,
            "--kitti_raw_dir", args.kitti_raw_dir, "--kitti_depth_dir", args.kitti_depth_dir,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  !! {key} exited with code {result.returncode} (crashed or errored) -- "
                  f"continuing with remaining models, you keep everything run so far.")
            crashed.append(key)

    # Merge every partial_<dataset>_<key>.json this run produced (works even
    # if you re-run just the models that crashed later and merge again).
    merged = {}
    for dataset_name in (["nyu", "kitti"] if args.dataset == "both" else [args.dataset]):
        for f in out_dir.glob(f"partial_{dataset_name}_*.json"):
            key = f.stem.replace(f"partial_{dataset_name}_", "")
            with open(f) as fh:
                merged[key] = json.load(fh)
        save_json(merged, str(out_dir / f"quant_results_{dataset_name}_merged.json"))
        print(f"\nMerged {len(merged)} model results -> {out_dir / f'quant_results_{dataset_name}_merged.json'}")

    if crashed:
        print(f"\nModels that crashed/errored this run: {crashed}")
        print("Re-run with --only <those keys> once, individually, to retry just them.")


if __name__ == "__main__":
    main()
