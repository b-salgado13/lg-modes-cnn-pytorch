"""Custom 3-layer convolutional network, equivalent to Appendix 3.

Mirrors the MATLAB `layers` array:

    convolution2dLayer(3,8,'Padding','same') -> batchNorm -> relu -> maxPool(2,2)
    convolution2dLayer(3,8,'Padding','same') -> batchNorm -> relu -> maxPool(2,2)
    convolution2dLayer(3,8,'Padding','same') -> batchNorm -> relu
    fullyConnectedLayer(numClasses) -> softmax -> classificationLayer

As in :class:`~src.models.simple_mlp.SimpleMLP`, the softmax is left
implicit and the model should be trained with `nn.CrossEntropyLoss`.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int, pool: bool) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 'same' padding
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """3-layer CNN with 8 filters per layer (Appendix 3).

    Args:
        num_classes: number of output classes.
        in_channels: 1 for grayscale input (matches the 128x192x1 input
            layer used in the thesis).
        input_size: ``(height, width)`` of the input images, used only to
            infer the flattened feature dimension for the final fully
            connected layer.
        filters: number of filters per convolutional layer (8 in the
            original network).
    """

    def __init__(
        self,
        num_classes: int = 19,
        in_channels: int = 1,
        input_size: Tuple[int, int] = (128, 192),
        filters: int = 8,
    ):
        super().__init__()
        self.block1 = _conv_block(in_channels, filters, pool=True)
        self.block2 = _conv_block(filters, filters, pool=True)
        self.block3 = _conv_block(filters, filters, pool=False)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)
            flat_dim = self._forward_features(dummy).flatten(1).shape[1]

        self.classifier = nn.Linear(flat_dim, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
