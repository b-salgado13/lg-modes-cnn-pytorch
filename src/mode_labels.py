"""Utilities to parse and describe Laguerre-Gauss mode labels.

The original MATLAB dataset stores images in folders named after the mode
they correspond to (see Section 3.2 of the thesis):

- ``m__<l>``     -> fiber-propagated mode with azimuthal number ``l`` (p = 0).
                    e.g. ``m__-6`` is l = -6, ``m__6`` is l = 6.
- ``LG_<l><p>``  -> free-space mode with azimuthal number ``l`` (+-1) and
                    radial number ``p`` (0, 3 or 5), sign of ``l`` written
                    directly before ``p``.
                    e.g. ``LG_-15`` is l = -1, p = 5; ``LG_13`` is l = 1, p = 3.

This module turns a folder/class name into a small ``ModeLabel`` object and
back into a human readable description, mirroring the logic implemented in
``PredictLGModeButtonPushed`` and ``SelectPatternButtonValueChanged`` in the
original App Designer GUI (Appendix 6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FIBER_RE = re.compile(r"^m__(-?\d+)$")
_FREE_SPACE_RE = re.compile(r"^LG_(-1|1)(\d+)$")


@dataclass(frozen=True)
class ModeLabel:
    """A single Laguerre-Gauss mode label.

    Attributes:
        medium: ``"fiber"`` for modes propagated through multimode optical
            fiber, ``"free_space"`` for modes propagated through free space.
        l: azimuthal (topological charge) quantum number.
        p: radial quantum number (always 0 for fiber-propagated modes in
            this dataset).
    """

    medium: str
    l: int
    p: int

    def describe(self) -> str:
        if self.medium == "fiber":
            return f"Optic Fiber: l = {self.l}, p = 0"
        return f"Free space: l = {self.l}, p = {self.p}"


def parse_folder_name(folder_name: str) -> ModeLabel:
    """Parse a dataset folder/class name into a :class:`ModeLabel`.

    Raises:
        ValueError: if ``folder_name`` does not match either naming
            convention (``m__<l>`` or ``LG_<l><p>``).
    """
    match = _FIBER_RE.match(folder_name)
    if match:
        return ModeLabel(medium="fiber", l=int(match.group(1)), p=0)

    match = _FREE_SPACE_RE.match(folder_name)
    if match:
        return ModeLabel(medium="free_space", l=int(match.group(1)), p=int(match.group(2)))

    raise ValueError(
        f"Folder name '{folder_name}' does not match the expected naming "
        "convention (m__<l> for fiber modes or LG_<l><p> for free-space "
        "modes). See src/mode_labels.py for details."
    )


def describe_class_name(class_name: str) -> str:
    """Convenience wrapper: parse + describe in a single call."""
    return parse_folder_name(class_name).describe()
