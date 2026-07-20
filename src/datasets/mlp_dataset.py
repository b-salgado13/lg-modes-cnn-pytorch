"""Preprocessing pipeline for the simple MLP (Appendix 1).

Reproduces, image by image, the loop in `apendice1_simple_mlp.m`:

1. Load a raw ``.jpg`` photograph.
2. Convert to grayscale.
3. Crop a fixed region that discards the black borders of the photograph
   (``imcrop(Im, [x y width height])`` in MATLAB).
4. Downsample by block-averaging (see :mod:`src.datasets.downsampling`).
5. Flatten into a single feature vector.

Only folders following the fiber-mode naming convention (``m__<l>``) are
used here, exactly as in the original Appendix 1 script, which loops over
``i = -9:9`` and only reads from ``.../Fotos/m__<i>`` folders.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.datasets.downsampling import down_sampling
from src.mode_labels import parse_folder_name

DEFAULT_CROP_BOX = (1900, 1100, 2899, 2499)  # (x, y, width, height), MATLAB convention


@dataclass
class MLPDataset:
    X: np.ndarray  # shape (n_samples, n_features)
    y: np.ndarray  # shape (n_samples,), integer class indices
    class_to_idx: Dict[str, int]


def _preprocess_image(
    path: Path,
    crop_box: Tuple[int, int, int, int],
    down_square: int,
) -> np.ndarray:
    x, y, width, height = crop_box
    with Image.open(path) as img:
        gray = img.convert("L")
        # PIL crop uses (left, upper, right, lower); MATLAB imcrop uses
        # (x, y, width, height) -> right = x + width, lower = y + height.
        cropped = gray.crop((x, y, x + width, y + height))
        array = np.asarray(cropped, dtype=np.float64)
    return down_sampling(array, down_square)


def build_mlp_dataset(
    data_dir: str | Path,
    crop_box: Tuple[int, int, int, int] = DEFAULT_CROP_BOX,
    down_square: int = 20,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".JPG", ".JPEG"),
) -> MLPDataset:
    """Build the (X, y) feature matrix used to train :class:`SimpleMLP`.

    Args:
        data_dir: directory containing one ``m__<l>`` subfolder per fiber
            mode (see the thesis' Section 3.2 for the naming convention).
        crop_box: ``(x, y, width, height)`` crop applied before
            downsampling, in MATLAB's ``imcrop`` convention.
        down_square: side length of the block-average downsampling window.
    """
    data_dir = Path(data_dir)
    class_dirs: List[Path] = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("m__")
    )
    if not class_dirs:
        raise FileNotFoundError(
            f"No 'm__<l>' folders found under {data_dir}. Appendix 1 only "
            "uses fiber-propagated modes; check the dataset layout in the "
            "README."
        )

    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

    features: List[np.ndarray] = []
    labels: List[int] = []
    for class_dir in class_dirs:
        # Validate the folder name early so mistakes are caught before
        # spending time preprocessing every image inside it.
        parse_folder_name(class_dir.name)
        label_idx = class_to_idx[class_dir.name]
        image_paths = sorted(
            p for p in class_dir.iterdir() if p.suffix in extensions
        )
        for image_path in image_paths:
            down = _preprocess_image(image_path, crop_box, down_square)
            features.append(down.reshape(-1))
            labels.append(label_idx)

    if not features:
        raise FileNotFoundError(f"No images found under {data_dir}.")

    X = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return MLPDataset(X=X, y=y, class_to_idx=class_to_idx)
