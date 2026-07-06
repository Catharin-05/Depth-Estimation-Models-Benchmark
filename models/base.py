"""
Common interface every model wrapper in this folder implements, so
run_benchmark.py can treat all 10 models identically.

    wrapper = load_model()                  # -> DepthModel
    depth, meta = wrapper.predict(pil_image) # depth: HxW float32 numpy array

`meta` always contains:
    is_metric   : True if depth is in meters (absolute scale)
    is_relative : True if depth is only affine-invariant (unknown scale/shift)
    larger_is_closer : convention used by this particular model's raw output
"""

from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
from PIL import Image


@dataclass
class DepthModel:
    name: str
    params_M: float          # from official docs/repo, or measured via count_params()
    is_metric: bool
    source: str               # "transformers" | "torch.hub" | "github" | "diffusers"
    predict_fn: Callable[[Image.Image], np.ndarray]
    larger_is_closer: bool = False
    notes: str = ""

    def predict(self, image: Image.Image):
        depth = self.predict_fn(image)
        meta = {
            "is_metric": self.is_metric,
            "is_relative": not self.is_metric,
            "larger_is_closer": self.larger_is_closer,
        }
        return np.asarray(depth, dtype=np.float32), meta
