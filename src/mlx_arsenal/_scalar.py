"""Narrowing helpers for ``mx.array.item()``.

MLX types ``array.item()`` as ``int | float | bool | complex`` (the ``scalar``
alias gained ``complex`` in mlx 0.32.1), so ``float(a.item())`` no longer
type-checks even when the array is known to hold a real dtype. These helpers
narrow the union once, in one place, instead of scattering casts.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["item_float", "item_int"]


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
