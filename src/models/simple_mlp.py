"""Simple multilayer perceptron, equivalent to Appendix 1's `patternnet(5)`.

MATLAB's `patternnet` builds a feed-forward network with one hidden layer
(``tansig`` activation, 5 neurons by default here) followed by a
``softmax`` output layer, trained for classification with
`Net.divideParam` splitting the data 70/15/15 into train/validation/test.

Here the softmax is left implicit: the model outputs raw logits and is
meant to be trained with `nn.CrossEntropyLoss`, which combines
`log_softmax` + negative log-likelihood internally.
"""
from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """Single hidden-layer perceptron for classification.

    Args:
        input_dim: number of input features (flattened, downsampled image).
        hidden_units: number of neurons in the hidden layer (5 in the
            original `patternnet(5)`).
        num_classes: number of output classes.
    """

    def __init__(self, input_dim: int, hidden_units: int = 5, num_classes: int = 19):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_units)
        self.activation = nn.Tanh()  # equivalent to MATLAB's default 'tansig'
        self.output = nn.Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.hidden(x))
        return self.output(x)
