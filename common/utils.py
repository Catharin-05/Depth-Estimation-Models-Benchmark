"""
Shared utilities for the monocular depth estimation shootout.

Covers three things every model script needs:
  1. count_params()          -> exact parameter count for any nn.Module
  2. benchmark_latency()     -> warmup + repeated-timing latency benchmark
  3. depth_to_pointcloud()   -> back-project a depth map to a 3D point cloud
  4. save/plot helpers for depth maps and point clouds

Design notes
------------
- All functions are backend-agnostic: they take plain numpy arrays / torch
  tensors, not model-specific objects, so the same code works for every
  model in models/*.py.
- depth_to_pointcloud() supports both:
    (a) metric depth (meters) with known/estimated camera intrinsics
    (b) relative / affine-invariant depth (Depth Anything, MiDaS, Marigold),
        in which case the "3D scene" is only correct up to an unknown
        global scale + shift -- this is called out explicitly wherever used.
"""

from __future__ import annotations
import time
import json
import numpy as np

try:
    import torch
except ImportError:
    torch = None


# --------------------------------------------------------------------------
# 1. Parameter counting
# --------------------------------------------------------------------------

def count_params(model) -> dict:
    """Return total and trainable parameter counts for a torch.nn.Module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": round(total / 1e6, 2),
    }


# --------------------------------------------------------------------------
# 2. Latency benchmarking
# --------------------------------------------------------------------------

def benchmark_latency(fn, n_warmup: int = 3, n_runs: int = 10, device: str = "cpu") -> dict:
    """
    Generic latency benchmark for any zero-arg callable `fn` (e.g. a closure
    that runs model(input) under torch.no_grad()).

    Returns mean/std/min/max in milliseconds over n_runs, after n_warmup
    untimed warmup calls (important for CPU: first call pays for kernel /
    thread-pool setup and is not representative).
    """
    if torch is not None and device == "cuda" and torch.cuda.is_available():
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None

    for _ in range(n_warmup):
        fn()
        sync()

    times_ms = []
    for _ in range(n_runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms = np.array(times_ms)
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "min_ms": float(times_ms.min()),
        "max_ms": float(times_ms.max()),
        "n_runs": n_runs,
        "device": device,
        "all_ms": times_ms.tolist(),
    }


# --------------------------------------------------------------------------
# 3. Depth -> 3D point cloud
# --------------------------------------------------------------------------

def estimate_intrinsics(width: int, height: int, hfov_deg: float = 60.0) -> dict:
    """
    Rough pinhole intrinsics estimate when the real camera/EXIF focal length
    is unknown (true for the AI-generated blog images here). Assumes a
    `hfov_deg` horizontal field of view (60 deg is a reasonable default for
    a "normal" ~35mm-equivalent lens) and a centered principal point.

    If you have real EXIF focal length + sensor size, replace this with the
    actual fx, fy, cx, cy -- reconstructions will be metrically meaningful
    instead of just "plausible".
    """
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    fy = fx  # assume square pixels
    cx, cy = width / 2.0, height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict,
    is_relative: bool = False,
    stride: int = 4,
    max_depth: float | None = None,
):
    """
    Back-project a dense depth map to a colored 3D point cloud using the
    standard pinhole model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z

    Parameters
    ----------
    depth : HxW array. Metric depth in meters, OR relative/inverse depth if
        is_relative=True (in which case Z is just the (normalized) model
        output and X, Y are only correct up to that same unknown scale --
        fine for visual inspection, not for measurement).
    rgb   : HxWx3 uint8 array, same resolution as depth.
    intrinsics : dict with fx, fy, cx, cy (see estimate_intrinsics).
    stride : subsample every `stride` pixels to keep point clouds small
        enough to render/save quickly.
    max_depth : optional clip to drop far-away / sky / invalid points.

    Returns
    -------
    points : (N, 3) float32 array
    colors : (N, 3) uint8 array
    """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    zs = depth[::stride, ::stride].astype(np.float32)

    if is_relative:
        # Depth Anything / MiDaS / Marigold output affine-invariant
        # "disparity-like" values where LARGER = CLOSER, i.e. true depth is
        # only known up to depth ~ 1/(a*disparity + b) for unknown a, b.
        #
        # IMPORTANT: don't do a literal 1/(x + tiny_eps) on the raw values.
        # Near-zero disparity (sky, far background -- often a big fraction
        # of the image) blows up toward the same huge asymptote for every
        # such pixel, so they all collapse onto one flat plane and swamp the
        # real structure (this was the original bug).
        #
        # Fix: robustly normalize disparity to [0, 1] first (percentile
        # clipping kills outlier pixels), THEN invert with an epsilon that's
        # scaled to that normalized range so the blow-up is bounded.
        lo, hi = np.percentile(zs, 1), np.percentile(zs, 99)
        d_norm = np.clip((zs - lo) / (hi - lo + 1e-8), 0.0, 1.0)  # 0=far, 1=near
        eps = 0.05  # bounds max "perspective stretch" to 1/eps = 20x
        zs = 1.0 / (d_norm + eps)
        zs = zs / zs.max() * 5.0  # rescale so nearest ~ small Z, farthest ~5 "units"
    else:
        # Metric models can still have a handful of wild outlier pixels
        # (e.g. sky mis-estimated as very far); clip to the 1st-99th
        # percentile so a couple of stray points don't stretch every axis.
        lo, hi = np.percentile(zs, 1), np.percentile(zs, 99)
        zs = np.clip(zs, lo, hi)

    valid = np.ones_like(zs, dtype=bool)
    if max_depth is not None:
        valid &= zs < max_depth
    valid &= zs > 0

    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy

    points = np.stack([xs[valid], -ys[valid], zs[valid]], axis=-1).astype(np.float32)
    colors = rgb[::stride, ::stride][valid]
    return points, colors


def save_pointcloud_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Write an ASCII .ply point cloud (viewable in MeshLab / CloudCompare /
    online .ply viewers, no extra dependencies needed)."""
    n = points.shape[0]
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def colorize_depth(depth: np.ndarray, cmap: str = "inferno"):
    """Normalize + colorize a depth map for visualization (no matplotlib
    dependency needed at call sites)."""
    import matplotlib.cm as cm

    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    colored = (cm.get_cmap(cmap)(d)[..., :3] * 255).astype(np.uint8)
    return colored


def plot_pointcloud_views(points: np.ndarray, colors: np.ndarray, out_path: str, title: str = ""):
    """Render a 3-panel figure (front, side, top view) of a point cloud with
    matplotlib's 3D scatter -- no open3d / plotly dependency required."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(15, 5))
    # NOTE on elev/azim: points are plotted as scatter(X, Z_depth, Y), i.e.
    # matplotlib's own (x, y, z) = (world X, world Z-depth, world Y-height).
    # elev is the angle above matplotlib's x-y plane (= world X / Z-depth
    # plane), so elev=90 looks straight down the world HEIGHT axis --
    # that's "top-down", not "front". A true "front" view (camera looking
    # along world depth, seeing X vs height) needs elev=0 with azim aimed
    # down matplotlib's y-axis (azim=-90). A "side" view (camera looking
    # along world X, seeing depth vs height) needs elev=0, azim=0.
    views = [("Front view", 0, -90), ("Side view", 0, 0), ("Top-down", 90, -90)]
    c = colors.astype(np.float32) / 255.0

    # Equal-aspect box so the cloud isn't visually squished when one axis
    # (e.g. depth) naturally spans a bigger range than X/Y.
    x_range = points[:, 0].max() - points[:, 0].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    box_aspect = (max(x_range, 1e-6), max(z_range, 1e-6), max(y_range, 1e-6))

    for i, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=c, s=0.5, marker=".")
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect(box_aspect)  # matplotlib >= 3.3
        except AttributeError:
            pass
        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Z (depth)")
        ax.set_zlabel("Y")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
