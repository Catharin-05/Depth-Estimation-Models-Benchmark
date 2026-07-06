"""
LeReS (Yin et al., 2021) -- aim-uofa/AdelaiDepth
https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS

"Learning to Recover 3D Scene Shape from a Single Image" -- ResNeXt101/
ResNet50 encoder-decoder that predicts relative depth *and* a learned
point-cloud shift, explicitly designed to fix the common failure where
naive monocular reconstructions come out warped/curved ("banana effect")
compared to the true 3D scene shape. No official pip package or
`transformers`/`torch.hub` integration -- installed by cloning the repo
directly, which is also why we implement it as a subprocess-style wrapper
around their `test_shape.py` rather than a clean Python import.
"""
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
from .base import DepthModel


def load_model(repo_dir: str = "./cache/AdelaiDepth/LeReS", ckpt_path: str = "./cache/leres_ckpt/res101.pth") -> DepthModel:
    repo_dir = Path(repo_dir)
    assert repo_dir.exists(), (
        f"Clone https://github.com/aim-uofa/AdelaiDepth into {repo_dir} first "
        "(git clone https://github.com/aim-uofa/AdelaiDepth) and download res101.pth "
        "per their README (Google Drive link -- not fetchable from this sandbox)."
    )

    def predict_fn(image: Image.Image) -> np.ndarray:
        # LeReS ships as standalone inference scripts rather than an importable
        # model class -- the officially documented path is running their
        # `Minist_Test/tools/test_shape.py` CLI against a folder of images
        # and reading back the .png/.npy depth outputs it writes. Left as a
        # subprocess call here to match how the repo is actually meant to be used.
        tmp_in = Path("/tmp/leres_in.png")
        image.convert("RGB").save(tmp_in)
        subprocess.run(
            [
                "python", str(repo_dir / "Minist_Test/tools/test_shape.py"),
                "--load_ckpt", ckpt_path,
                "--input", str(tmp_in),
            ],
            check=True,
        )
        depth = np.load("/tmp/leres_in_depth.npy")  # path convention from their script
        return depth.astype(np.float32)

    return DepthModel(
        name="LeReS (ResNeXt101)",
        params_M=279.0,  # ResNeXt101 encoder + decoder, approx per repo config
        is_metric=False,
        source="github",
        larger_is_closer=False,
        predict_fn=predict_fn,
        notes="Relative depth with explicit point-cloud shift correction for scene-shape "
              "recovery. CLI-based, not a clean importable API -- see docstring.",
    )
