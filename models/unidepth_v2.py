"""
UniDepth V2 (Piccinelli et al., 2024) -- lpiccinelli-eth/UniDepth
https://github.com/lpiccinelli-eth/UniDepth

Self-promptable camera-aware metric depth: jointly predicts depth *and*
camera intrinsics (or uses provided intrinsics if available), removing the
need for a separate focal-length estimation step. Backbone options include
ViT-S/B/L (DINOv2-based); the large ViT-L variant is the common default.
Not on the `transformers` hub -- installed as a pip package / via
torch.hub straight from the official repo.
"""
import numpy as np
from PIL import Image
import torch
from .base import DepthModel


def load_model(backbone: str = "vitl14") -> DepthModel:
    model = torch.hub.load(
        "lpiccinelli-eth/UniDepth", "UniDepth",
        version="v2", backbone=backbone, pretrained=True,
        trust_repo=True,
    )
    model.eval()

    def predict_fn(image: Image.Image) -> np.ndarray:
        rgb = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float()
        with torch.no_grad():
            # intrinsics=None -> model estimates its own camera model internally
            predictions = model.infer(rgb)
        depth = predictions["depth"].squeeze().cpu().numpy()
        return depth

    params_M = {"vits14": 25.0, "vitb14": 98.0, "vitl14": 335.0}.get(backbone, -1)

    return DepthModel(
        name=f"UniDepth V2 ({backbone})",
        params_M=params_M,
        is_metric=True,
        source="torch.hub",
        larger_is_closer=False,
        predict_fn=predict_fn,
        notes="Metric depth (meters), self-estimates camera intrinsics if not supplied.",
    )
