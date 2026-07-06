"""
Depth Pro (Bochkovskii et al., Apple, 2024) -- apple/ml-depth-pro
https://huggingface.co/apple/DepthPro-hf | https://arxiv.org/abs/2410.02073

Multi-scale plain-ViT patch encoder (shared weights across scales) fused
via a DPT-style decoder, plus a separate focal-length estimation head so
it needs no camera metadata. Zero-shot *metric* depth with sharp
boundaries; paper reports 0.3s for a 2.25MP output on a V100 GPU.
~950M-1B total params (multiple ViT-L/16-scale encoders + decoder + focal
length head; exact count depends on counting the shared-weight patch
encoder once or per-scale -- see notes).
"""
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from .base import DepthModel


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint: str = "apple/DepthPro-hf", device: str | None = None,
               use_fov_model: bool = False) -> DepthModel:
    """
    use_fov_model=False (default here, NOT the library default) skips the
    extra Dinov2-based field-of-view estimation encoder. Two reasons:
      1. Speed: it roughly doubles the forward pass cost, and on CPU/MPS
         this was measured taking 100+ seconds/image with it enabled.
      2. Accuracy on datasets with unusual intrinsics: the FOV head predicts
         focal length from the image itself, and that estimate directly
         scales the final metric depth. On NYU's Kinect camera (quite
         different FOV than Depth Pro's typical training/eval photos), a
         bad focal-length guess propagates into every pixel -- likely why
         the first run showed AbsRel ~0.72 despite Depth Pro's usual
         zero-shot accuracy. Since is_metric=True skips our scale-alignment
         step entirely (correctly so for a "real" metric model), a wrong
         internal scale estimate has nowhere to be corrected.
    Set use_fov_model=True if you specifically want Depth Pro's self-
    estimated focal length (needed when you have NO other source of
    intrinsics); with it off, accuracy depends on how reasonable Depth
    Pro's built-in default FOV assumption is for your images.
    """
    device = device or _best_device()
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint, use_fov_model=use_fov_model)
    model.eval()
    model.to(device)

    def predict_fn(image: Image.Image) -> np.ndarray:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Official post-processing (not a manual resize) -- matches how
        # Depth Anything V2 / DPT are handled via the `pipeline` wrapper,
        # so all HF-based models in this benchmark go through the same
        # library-recommended path.
        post = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)],
        )
        depth = post[0]["predicted_depth"]
        return depth.detach().cpu().numpy()

    return DepthModel(
        name="Depth Pro",
        params_M=952.0,  # approximate, see docstring
        is_metric=True,
        source="transformers",
        larger_is_closer=False,
        predict_fn=predict_fn,
        notes="Metric depth in meters. FOV/focal-length self-estimation disabled by "
              "default here (use_fov_model=False) for speed and to avoid a bad focal-length "
              "guess silently distorting metric scale on atypical cameras -- see docstring. "
              "Author-reported 0.3s @ 2.25MP on a V100 GPU; expect much slower on CPU/MPS.",
    )
