"""
Depth Anything V2 (Yang et al., 2024) -- DepthAnything/Depth-Anything-V2
https://huggingface.co/depth-anything

DINOv2 encoder + DPT decoder, trained with a teacher-student pipeline on
595K synthetic + 62M pseudo-labeled real images. Four sizes:
  Small (24.8M, Apache-2.0) | Base (97.5M) | Large (335.3M) | Giant (1.3B,
  not publicly released as of this writing). Base/Large are CC-BY-NC-4.0.
Also ships metric-depth fine-tunes (indoor/outdoor). Relative-depth
variant used here via the `transformers` pipeline (officially supported
since transformers 4.42+).
"""
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
from .base import DepthModel

CHECKPOINTS = {
    "small": ("depth-anything/Depth-Anything-V2-Small-hf", 24.8),
    "base": ("depth-anything/Depth-Anything-V2-Base-hf", 97.5),
    "large": ("depth-anything/Depth-Anything-V2-Large-hf", 335.3),
}


def _best_device() -> int | str:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


def load_model(variant: str = "small", device=None) -> DepthModel:
    checkpoint, params_M = CHECKPOINTS[variant]
    device = _best_device() if device is None else device
    pipe = pipeline(task="depth-estimation", model=checkpoint, device=device)

    def predict_fn(image: Image.Image) -> np.ndarray:
        out = pipe(image)
        return np.array(out["predicted_depth"])

    return DepthModel(
        name=f"Depth Anything V2 ({variant})",
        params_M=params_M,
        is_metric=False,
        source="transformers",
        larger_is_closer=True,
        predict_fn=predict_fn,
        notes="Relative depth (disparity-like, larger=closer). DINOv2+DPT.",
    )
