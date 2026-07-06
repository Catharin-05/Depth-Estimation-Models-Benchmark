"""
MiDaS (Ranftl et al., 2019/2021) -- intel-isl/MiDaS
https://github.com/isl-org/MiDaS

Relative (affine-invariant inverse) depth. Loaded via torch.hub, weights
come from GitHub releases (isl-org/MiDaS/releases). Requires `timm`.

Variants: DPT_BEiT_L_512 (~345M), DPT_Large (~344M), DPT_Hybrid (~123M),
MiDaS_small (~21M, our own measured count -- see blog for methodology).
"""
import torch
import numpy as np
from PIL import Image
from .base import DepthModel

CHECKPOINTS = {
    "MiDaS_small": 21.3,
    "DPT_Hybrid": 123.1,
    "DPT_Large": 344.1,
}


def load_model(variant: str = "DPT_Hybrid") -> DepthModel:
    model = torch.hub.load("intel-isl/MiDaS", variant, trust_repo=True)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.dpt_transform if "DPT" in variant else transforms.small_transform

    def predict_fn(image: Image.Image) -> np.ndarray:
        img = np.array(image.convert("RGB"))
        inp = transform(img).unsqueeze(0) if transform(img).dim() == 3 else transform(img)
        with torch.no_grad():
            pred = model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        return pred.cpu().numpy()

    return DepthModel(
        name=f"MiDaS ({variant})",
        params_M=CHECKPOINTS.get(variant, -1),
        is_metric=False,
        source="torch.hub",
        larger_is_closer=True,  # MiDaS outputs inverse depth: bigger = nearer
        predict_fn=predict_fn,
        notes="Affine-invariant relative depth. Classic MiDaS v3.1 release.",
    )
