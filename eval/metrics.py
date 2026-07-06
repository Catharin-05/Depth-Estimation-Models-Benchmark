"""
Standard monocular depth evaluation metrics -- the same ones reported in
the MiDaS, ZoeDepth, Depth Anything V2, UniDepth V2, Metric3D v2 and Depth
Pro papers, so your numbers land in the same units/scale as their tables.

Metrics:
    AbsRel  = mean(|pred - gt| / gt)              -- lower is better
    RMSE    = sqrt(mean((pred - gt)^2))            -- lower is better
    log10   = mean(|log10(pred) - log10(gt)|)      -- lower is better
    delta1  = % of pixels with max(pred/gt, gt/pred) < 1.25    -- higher is better
    delta2  = ... < 1.25^2                                      -- higher is better
    delta3  = ... < 1.25^3                                      -- higher is better

Relative-depth models (MiDaS, DPT, Depth Anything V2 base checkpoints,
Marigold, LeReS) are only affine-invariant: pred ~ a*depth_true + b for
UNKNOWN a, b. Comparing them to metric ground truth directly is unfair and
meaningless -- the standard protocol (used in every paper above) is to
first solve for the best-fit a, b per image via least squares, apply it,
THEN compute the metrics. `align_scale_shift()` below does exactly that.
Metric models (ZoeDepth, UniDepth V2, Metric3D v2, Depth Pro) skip this
step entirely -- their raw output is compared directly.
"""
import numpy as np


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Least-squares solve for scale `a` and shift `b` minimizing
    ||a*pred + b - gt||^2 over valid pixels, then return a*pred + b.
    This is the standard alignment for affine-invariant relative depth
    (see MiDaS/Marigold/Depth Anything eval protocols).
    """
    p = pred[valid_mask].astype(np.float64)
    g = gt[valid_mask].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, g, rcond=None)
    return pred * a + b


def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray, max_depth: float, min_depth: float = 1e-3) -> dict:
    """
    pred, gt: HxW arrays, same shape, in meters (pred should already be
    metric-aligned -- see align_scale_shift for relative-depth models).
    """
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(pred) & (pred > min_depth)
    if valid.sum() < 10:
        return {"error": "too few valid pixels", "n_valid": int(valid.sum())}

    p = pred[valid]
    g = gt[valid]

    abs_rel = np.mean(np.abs(p - g) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    log10_err = np.mean(np.abs(np.log10(p) - np.log10(g)))

    ratio = np.maximum(p / g, g / p)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return {
        "AbsRel": float(abs_rel),
        "RMSE": float(rmse),
        "log10": float(log10_err),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
        "n_valid": int(valid.sum()),
    }


def evaluate_prediction(pred: np.ndarray, gt: np.ndarray, is_metric: bool, max_depth: float,
                         larger_is_closer: bool = False, min_depth: float = 1e-3) -> dict:
    """
    Full pipeline: align if the model isn't metric, then compute standard
    metrics.

    Two different relative-depth conventions need two different alignment
    strategies:
      - larger_is_closer=True  (MiDaS/Depth-Anything-style disparity):
            disparity ~= a * (1/depth_true) + b   (affine in INVERSE depth)
        so we align `pred` against `1/gt` in disparity space, THEN invert
        the aligned result back to depth. Aligning in depth space directly
        (i.e. naively inverting pred first) does NOT recover the right
        answer, since an affine map in disparity space is not affine in
        depth space.
      - larger_is_closer=False (some models output relative depth
        directly, larger=farther): align_scale_shift in depth space as-is.
    """
    pred = pred.astype(np.float64).copy()

    if not is_metric:
        valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(pred)
        if larger_is_closer:
            disparity_gt = 1.0 / np.clip(gt, min_depth, None)
            aligned_disparity = align_scale_shift(pred, disparity_gt, valid)
            pred = 1.0 / np.clip(aligned_disparity, 1e-6, None)
        else:
            pred = align_scale_shift(pred, gt, valid)

    return compute_depth_metrics(pred, gt, max_depth, min_depth)


def aggregate_metrics(per_image_metrics: list[dict]) -> dict:
    """Average metrics across all images for one model (skipping any that errored)."""
    keys = ["AbsRel", "RMSE", "log10", "delta1", "delta2", "delta3"]
    valid_rows = [m for m in per_image_metrics if "error" not in m]
    if not valid_rows:
        return {"error": "no valid images"}
    return {k: float(np.mean([m[k] for m in valid_rows])) for k in keys} | {"n_images": len(valid_rows)}
