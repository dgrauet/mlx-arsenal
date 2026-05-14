"""State-dict container types."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

__all__ = ["StateDict"]


@dataclass(frozen=True)
class StateDict:
    """A loaded model state-dict with memory + dtype accounting.

    Attributes:
        sd: Parameter name → array.
        size: Total memory footprint in bytes (sum of ``array.nbytes``).
        dtype: Set of distinct dtypes present (useful to detect mixed-
            precision shards).
    """

    sd: dict[str, mx.array]
    size: int
    dtype: set[mx.Dtype]

    def footprint(self) -> int:
        """Total memory footprint in bytes."""
        return self.size
