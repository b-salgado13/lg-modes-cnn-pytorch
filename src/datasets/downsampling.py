"""Block-average downsampling, equivalent to Appendix 2 (`DownSampling_v2.m`).

The original MATLAB function slides a non-overlapping ``SquareSize`` x
``SquareSize`` window over a 2D field and replaces each window with its
average value, discarding any remainder rows/columns that do not fit a full
block (exactly like the MATLAB code, which computes ``Limit`` via integer
division and simply ignores the leftover ``res`` pixels).
"""
from __future__ import annotations

import numpy as np


def down_sampling(field: np.ndarray, square_size: int) -> np.ndarray:
    """Vectorized block-average downsampling.

    Args:
        field: 2D array (grayscale image) to downsample.
        square_size: side length, in pixels, of the averaging window.

    Returns:
        A 2D array of shape ``(rows // square_size, cols // square_size)``.
    """
    field = np.asarray(field, dtype=np.float64)
    rows, cols = field.shape[:2]
    limit_rows = rows // square_size
    limit_cols = cols // square_size

    if limit_rows == 0 or limit_cols == 0:
        raise ValueError(
            f"square_size={square_size} is larger than the input field "
            f"({rows}x{cols}); cannot form a single block."
        )

    # Drop the remainder pixels, exactly like MATLAB's `res`/`res2` handling.
    trimmed = field[: limit_rows * square_size, : limit_cols * square_size]
    blocks = trimmed.reshape(limit_rows, square_size, limit_cols, square_size)
    return blocks.mean(axis=(1, 3))


def down_sampling_reference(field: np.ndarray, square_size: int) -> np.ndarray:
    """Literal, loop-based translation of `DownSampling_v2.m`.

    Slower than :func:`down_sampling`, but kept as a 1:1 reference
    implementation and for unit-testing the vectorized version above.
    """
    field = np.asarray(field, dtype=np.float64)
    rows, cols = field.shape[:2]
    limit_rows = rows // square_size
    limit_cols = cols // square_size

    out = np.zeros((limit_rows, limit_cols), dtype=np.float64)
    for m in range(limit_rows):
        for p in range(limit_cols):
            ini_m, fin_m = m * square_size, (m + 1) * square_size
            ini_n, fin_n = p * square_size, (p + 1) * square_size
            block = field[ini_m:fin_m, ini_n:fin_n]
            out[m, p] = block.sum() / (square_size ** 2)
    return out
