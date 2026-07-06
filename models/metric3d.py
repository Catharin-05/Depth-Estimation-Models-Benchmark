"""
Metric3D v2 (Hu / Yin et al., 2024) -- YvanYin/Metric3D
https://github.com/YvanYin/Metric3D

Canonical-camera-space transform lets a single model give zero-shot
*metric* depth and surface normals across wildly different camera
intrinsics -- pass in real intrinsics if you have them, otherwise the
repo's utilities estimate a plausible default. ConvNeXt-L or ViT-L/g
backbones depending on checkpoint.
"""
import numpy as np
from PIL import Image
import torch
from .base import DepthModel

CHECKPOINTS = {
    "convnext_large": ("metric3d_convnext_large", 122.0),
    "vit_large": ("metric3d_vit_large", 335.0),
    "vit_giant2": ("metric3d_vit_giant2", 1300.0),
}


def load_model(variant: str = "vit_large", intrinsics=None) -> DepthModel:
    hub_name, params_M = CHECKPOINTS[variant]
    model = torch.hub.load("yvanyin/metric3d", hub_name, pretrain=True, trust_repo=True)
    model.eval()

    def predict_fn(image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        # Metric3D needs approximate camera intrinsics [fx, fy, cx, cy];
        # fall back to a 60deg-HFOV estimate if the caller didn't supply real ones.
        h, w = rgb.shape[:2]
        intr = intrinsics or {
            "fx": w / (2 * np.tan(np.radians(30))), "fy": w / (2 * np.tan(np.radians(30))),
            "cx": w / 2, "cy": h / 2,
        }
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference(
                {"input": rgb, "intrinsic": [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]}
            )
        return pred_depth.squeeze().cpu().numpy()

    return DepthModel(
        name=f"Metric3D v2 ({variant})",
        params_M=params_M,
        is_metric=True,
        source="torch.hub",
        larger_is_closer=False,
        predict_fn=predict_fn,
        notes="Metric depth (meters); accuracy depends on how close the supplied "
              "intrinsics are to the real camera -- a real limitation for AI-generated images "
              "with no EXIF data.",
    )
