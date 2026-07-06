"""
Marigold (Ke et al., 2024) -- prs-eth/marigold
https://huggingface.co/prs-eth

Repurposes a frozen Stable-Diffusion latent-diffusion backbone (~865M-1B
params, U-Net + VAE) for affine-invariant relative depth by treating depth
maps as "images" to denoise, conditioned on the RGB input. Very high
quality but computationally heavy (multi-step diffusion sampling per
image); the LCM distilled variant used below trades a little quality for
~10x fewer denoising steps. Installed via `diffusers`.
"""
import numpy as np
from PIL import Image
import torch
from diffusers import MarigoldDepthPipeline
from .base import DepthModel


def load_model(checkpoint: str = "prs-eth/marigold-depth-lcm-v1-0", num_steps: int = 4) -> DepthModel:
    pipe = MarigoldDepthPipeline.from_pretrained(checkpoint, variant="fp16" if torch.cuda.is_available() else None)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def predict_fn(image: Image.Image) -> np.ndarray:
        output = pipe(image, num_inference_steps=num_steps, ensemble_size=1)
        depth = output.prediction[0]  # affine-invariant relative depth, normalized [0,1]-ish
        return np.asarray(depth, dtype=np.float32).squeeze()

    return DepthModel(
        name="Marigold (LCM, 4-step)",
        params_M=865.0,  # Stable Diffusion 2 U-Net + VAE, approx
        is_metric=False,
        source="diffusers",
        larger_is_closer=True,
        predict_fn=predict_fn,
        notes="Diffusion-based relative depth. Much slower than regression models "
              "even distilled (4 denoising steps vs 1 forward pass) -- this is the "
              "central latency story of this shootout.",
    )
