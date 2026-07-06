"""
DPT -- Dense Prediction Transformer (Ranftl et al., 2021)
https://huggingface.co/Intel/dpt-large

This is the standalone "DPT" checkpoint (ViT-Large encoder, trained on the
MiDaS mixed dataset) as distinct from the later MiDaS v3.1 releases in
midas.py -- included separately because DPT is commonly cited as its own
baseline in depth-estimation literature. Loaded via the `transformers`
depth-estimation pipeline, which is the officially recommended usage.

~344M params (ViT-L/16 backbone + DPT decoder head). Relative depth.
"""
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
from .base import DepthModel


def _best_device() -> int | str:
    # transformers pipeline device arg: -1 = cpu, 0 = cuda:0, "mps" = Apple Silicon
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


def load_model(checkpoint: str = "Intel/dpt-large", device=None) -> DepthModel:
    device = _best_device() if device is None else device
    pipe = pipeline(task="depth-estimation", model=checkpoint, device=device)

    def predict_fn(image: Image.Image) -> np.ndarray:
        out = pipe(image)
        return np.array(out["predicted_depth"]) if "predicted_depth" in out else np.array(out["depth"])

    return DepthModel(
        name="DPT-Large",
        params_M=343.0,
        is_metric=False,
        source="transformers",
        larger_is_closer=False,
        predict_fn=predict_fn,
        notes="Original DPT baseline (ViT-L/16), relative depth, MiDaS-style training mix.",
    )
