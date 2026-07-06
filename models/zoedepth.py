"""
ZoeDepth (Bhat et al., 2023) -- isl-org/ZoeDepth
https://github.com/isl-org/ZoeDepth

Combines a MiDaS-style relative depth backbone with metric "bin" heads
fine-tuned per-domain (NYU indoor / KITTI outdoor), giving zero-shot
*metric* depth in meters. ~345M params with the BEiT384-L backbone (the
paper also reports 112M/102M/42M variants with lighter backbones -- see
blog comparison table).
"""
import torch
import numpy as np
from PIL import Image
from .base import DepthModel


def _patch_timm_block_compat():
    """
    ZoeDepth's code calls block.drop_path(x) directly, matching an older
    timm `Block` class. Newer timm split this into drop_path1/drop_path2
    (separate paths for attention vs MLP), so that attribute no longer
    exists -- same root cause as the state-dict key mismatch, just showing
    up at inference time instead of load time. Alias drop_path -> drop_path1
    on timm's Block class so old code calling it directly still works.
    """
    try:
        import timm.models.vision_transformer as vit
        if not hasattr(vit.Block, "drop_path") and hasattr(vit.Block, "drop_path1"):
            vit.Block.drop_path = property(lambda self: self.drop_path1)
    except Exception:
        pass  # best-effort; if timm's internals differ, the original error will resurface


def load_model(variant: str = "ZoeD_NK") -> DepthModel:
    _patch_timm_block_compat()
    # ZoeD_N: NYU-indoor only | ZoeD_K: KITTI-outdoor only | ZoeD_NK: both (recommended default)
    #
    # The official checkpoint was saved against an older `timm` version that
    # registered some BEiT relative-position-index buffers the current
    # `timm` doesn't create the same way -- causes a strict state_dict load
    # to fail on "unexpected keys" even though the actual model weights are
    # fine. Patch torch's load_state_dict to be non-strict just for this
    # one call (restored immediately after), rather than requiring an old
    # timm version system-wide.
    _orig_load_state_dict = torch.nn.Module.load_state_dict
    def _non_strict_load(self, state_dict, strict=True, **kwargs):
        return _orig_load_state_dict(self, state_dict, strict=False, **kwargs)
    torch.nn.Module.load_state_dict = _non_strict_load
    try:
        model = torch.hub.load("isl-org/ZoeDepth", variant, pretrained=True, trust_repo=True)
    finally:
        torch.nn.Module.load_state_dict = _orig_load_state_dict
    model.eval()

    def predict_fn(image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            depth = model.infer_pil(image.convert("RGB"))  # returns metric depth in meters
        return np.asarray(depth, dtype=np.float32)

    return DepthModel(
        name=f"ZoeDepth ({variant})",
        params_M=345.0,
        is_metric=True,
        source="torch.hub",
        larger_is_closer=False,
        predict_fn=predict_fn,
        notes="Metric depth (meters), MiDaS-BEiT-L backbone + metric bin heads.",
    )