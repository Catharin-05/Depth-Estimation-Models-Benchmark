"""
PatchFusion (Li et al., 2024) -- zhyever/PatchFusion
https://github.com/zhyever/PatchFusion

Not a new backbone -- a tiling / fusion *framework* wrapped around an
existing base depth model (ZoeDepth by default, DepthAnything supported
too) that runs the base model on high-res tiles/patches and fuses them
back into one consistent, much higher-resolution depth map. Because of
this, "parameter count" and "latency" are dominated by the wrapped base
model x (number of tiles), not a fixed architecture -- see notes.
"""
import numpy as np
from PIL import Image
import torch
from .base import DepthModel


def load_model(base_model: str = "zoedepth", patch_process_shape=(384, 512), tile_cfg="1-4") -> DepthModel:
    model = torch.hub.load(
        "zhyever/PatchFusion", "PatchFusion",
        base_model=base_model,
        trust_repo=True,
    )
    model.eval()

    def predict_fn(image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        with torch.no_grad():
            # mode "r" = regular grid tiling; see repo for "p16"/"p49" patch configs
            depth = model.infer_pil(rgb, mode="r", patch_process_shape=patch_process_shape, tile_cfg=tile_cfg)
        return np.asarray(depth, dtype=np.float32)

    return DepthModel(
        name=f"PatchFusion ({base_model} base, {tile_cfg} tiling)",
        params_M=345.0,  # dominated by the wrapped ZoeDepth base model's params
        is_metric=(base_model == "zoedepth"),
        source="torch.hub",
        larger_is_closer=False,
        predict_fn=predict_fn,
        notes="Tiled high-resolution fusion on top of a base model -- expect latency "
              "roughly proportional to number of tiles processed, not a single fwd pass.",
    )
