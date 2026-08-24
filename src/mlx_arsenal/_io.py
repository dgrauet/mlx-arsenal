"""Private I/O helpers shared across subpackages."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import mlx.core as mx


def load_tensor_dict(path: str | Path) -> dict[str, mx.array]:
    """``mx.load`` a safetensors file, guaranteeing a tensor dict."""
    loaded = mx.load(str(path))
    if not isinstance(loaded, dict):
        raise TypeError(
            f"expected dict from safetensors, got {type(loaded).__name__} for {str(path)!r}"
        )
    return cast(dict[str, mx.array], loaded)
