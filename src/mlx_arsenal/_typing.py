"""Workarounds for gaps in mlx's bundled type stubs.

Everything here exists only to satisfy the type checker; the runtime behaviour
is what plain mlx would do. Delete a helper once upstream makes it redundant.

- ``item_float`` / ``item_int``: mlx types ``array.item()`` as
  ``int | float | bool | complex`` (the ``scalar`` alias gained ``complex`` in
  0.32.1), so ``float(a.item())`` no longer type-checks even for real dtypes.
- ``array_from_any``: ``mx.array`` accepts a ``DLPackCompatible`` Protocol whose
  members are declared as mutable attributes, which ``np.ndarray`` does not
  satisfy structurally (ml-explore/mlx#4371). Whether a given call is rejected
  depends on the installed numpy's own stubs, so per-call ``ty: ignore``
  comments flip between "needed" and "unused" across the support matrix.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

__all__ = ["array_from_any", "item_float", "item_int"]


def item_float(a: mx.array) -> float:
    """Return the single element of ``a`` as a Python ``float``.

    Raises:
        TypeError: if ``a`` holds a complex value.
    """
    value = a.item()
    if isinstance(value, complex):
        raise TypeError(f"expected a real scalar, got complex {value!r}")
    return float(value)


def item_int(a: mx.array) -> int:
    """Return the single element of ``a`` as a Python ``int``.

    Raises:
        TypeError: if ``a`` holds a complex value.
    """
    value = a.item()
    if isinstance(value, complex):
        raise TypeError(f"expected a real scalar, got complex {value!r}")
    return int(value)


def array_from_any(values: Any, dtype: mx.Dtype | None = None) -> mx.array:
    """``mx.array(values, dtype)`` for a buffer mlx's stubs reject (e.g. numpy)."""
    return mx.array(values, dtype=dtype)
