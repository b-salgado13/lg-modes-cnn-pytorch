"""Transfer-learning backbones, equivalent to Appendices 4 (EfficientNet-B0)
and 5 (ResNet18 / ResNet50 / ResNet101).

The MATLAB scripts:

1. Load an ImageNet-pretrained network.
2. Replace its final learnable layer + classification layer with new ones
   sized for `numClasses`.
3. Freeze the first `flayer` layers (`freezeWeights`) so only the last
   blocks and the new head are fine-tuned.

Here, step 3 is reproduced by freezing a *fraction* of the backbone's
parameters (from the input side) rather than an exact layer count, since
MATLAB's `layerGraph` layer indices do not translate one-to-one to
`torchvision`'s module structure across versions. See the README for
details.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torch import nn
from torchvision import models

SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet18", "resnet50", "resnet101")


def _freeze_first_fraction(named_params, fraction: float) -> None:
    """Freeze the first `fraction` of an (ordered) list of parameters."""
    named_params = list(named_params)
    n_freeze = int(len(named_params) * fraction)
    for _, param in named_params[:n_freeze]:
        param.requires_grad = False


def build_efficientnet_b0(
    num_classes: int = 19,
    pretrained: bool = True,
    freeze_fraction: float = 0.7,
) -> nn.Module:
    """Build an EfficientNet-B0 classifier (Appendix 4).

    Args:
        num_classes: number of output classes.
        pretrained: whether to start from ImageNet weights.
        freeze_fraction: fraction (0-1) of the backbone's *feature*
            parameters (input side first) to freeze during fine-tuning.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    _freeze_first_fraction(model.features.named_parameters(), freeze_fraction)
    return model


_RESNET_BUILDERS: Dict[str, Tuple[Callable[..., nn.Module], object]] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
}


def build_resnet(
    variant: str = "resnet18",
    num_classes: int = 19,
    pretrained: bool = True,
    freeze_fraction: float = 0.7,
) -> nn.Module:
    """Build a ResNet classifier (Appendix 5).

    Args:
        variant: one of ``"resnet18"``, ``"resnet50"``, ``"resnet101"``
            (the thesis trained all three and compared results, ultimately
            keeping ResNet18 for the final GUI).
        num_classes: number of output classes.
        pretrained: whether to start from ImageNet weights.
        freeze_fraction: fraction (0-1) of the backbone's parameters
            (input side first, excluding the new `fc` head) to freeze.
    """
    if variant not in _RESNET_BUILDERS:
        raise ValueError(
            f"Unknown ResNet variant '{variant}'. Choose one of "
            f"{sorted(_RESNET_BUILDERS)}."
        )
    builder, weights_enum = _RESNET_BUILDERS[variant]
    model = builder(weights=weights_enum if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    backbone_params = [
        (name, param) for name, param in model.named_parameters() if not name.startswith("fc.")
    ]
    _freeze_first_fraction(backbone_params, freeze_fraction)
    return model


def build_model(
    backbone: str,
    num_classes: int = 19,
    pretrained: bool = True,
    freeze_fraction: float = 0.7,
) -> nn.Module:
    """Dispatch helper used by the training/prediction scripts."""
    if backbone == "efficientnet_b0":
        return build_efficientnet_b0(num_classes, pretrained, freeze_fraction)
    if backbone in _RESNET_BUILDERS:
        return build_resnet(backbone, num_classes, pretrained, freeze_fraction)
    raise ValueError(
        f"Unknown backbone '{backbone}'. Choose one of {SUPPORTED_BACKBONES}."
    )
