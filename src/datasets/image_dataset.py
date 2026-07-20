"""Dataset helpers for the CNN (Appendix 3) and transfer-learning (Appendices
4-5) networks.

Both rely on `torchvision.datasets.ImageFolder`, which mirrors MATLAB's
`imageDatastore(..., 'LabelSource', 'foldernames')`: one subfolder per class,
class names taken directly from the folder names, sorted alphabetically to
build `class_to_idx` (this ordering is saved alongside every checkpoint so
predictions can always be mapped back to a class name).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms


def build_transform(
    image_size: Tuple[int, int],
    channels: int,
    normalize_imagenet: bool = False,
) -> transforms.Compose:
    """Build the torchvision transform pipeline for a given image config.

    Args:
        image_size: ``(height, width)`` the network expects.
        channels: 1 for grayscale (simple CNN), 3 for RGB (transfer
            learning backbones).
        normalize_imagenet: if True, normalize with ImageNet mean/std
            (recommended when fine-tuning ImageNet-pretrained backbones).
    """
    ops = []
    if channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    else:
        ops.append(transforms.Lambda(lambda img: img.convert("RGB")))
    ops.append(transforms.Resize(image_size))
    ops.append(transforms.ToTensor())
    if normalize_imagenet:
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(ops)


def build_image_folder_dataset(
    data_dir: str | Path,
    image_size: Tuple[int, int],
    channels: int,
    normalize_imagenet: bool = False,
) -> datasets.ImageFolder:
    """Build an `ImageFolder` dataset with the appropriate transforms.

    Raises:
        FileNotFoundError: if `data_dir` does not contain any class
            subfolders (mirrors MATLAB erroring out on an empty
            `imageDatastore`).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    transform = build_transform(image_size, channels, normalize_imagenet)
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    if len(dataset.classes) == 0:
        raise FileNotFoundError(f"No class subfolders found under {data_dir}.")
    return dataset
