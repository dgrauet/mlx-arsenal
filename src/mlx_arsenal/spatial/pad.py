"""Padding operations.

Historical note: early MLX releases lacked an edge/replicate mode, so this
module hand-rolled it. ``mx.pad(mode="edge")`` exists now; ``replicate_pad``
is kept as a thin, torch-named wrapper.
"""

from collections.abc import Sequence

import mlx.core as mx


def replicate_pad(x: mx.array, pad_widths: Sequence[tuple[int, int]]) -> mx.array:
    """Pad a tensor by replicating edge values.

    Equivalent to ``torch.nn.functional.pad(x, ..., mode="replicate")`` /
    ``mx.pad(x, ..., mode="edge")``.

    Args:
        x: Input tensor of any shape.
        pad_widths: One ``(before, after)`` pair per dimension.
            Length must match ``x.ndim``.

    Returns:
        Padded tensor.
    """
    return mx.pad(x, [tuple(p) for p in pad_widths], mode="edge")
