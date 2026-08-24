"""Adaptive Layer Norm (AdaLN) modulation primitives.

The DiT family (and its many spin-offs) modulates per-block features with
shift/scale/gate parameters produced from a conditioning signal —
typically the timestep embedding, sometimes pooled text. Each block
consumes a tuple of chunks and applies them around its
``norm → attention/MLP → residual`` stages.

The pattern is small and uniform but every port re-implements it. This
module exposes the building blocks:

- :class:`AdaLNModulation` — ``SiLU → Linear → split into N chunks``,
  the projection that turns a conditioning vector into modulation
  parameters. Supports any chunk count (1, 2, 4, 6, 9 are the common
  ones — single gate, scale+shift, AV cross-attn, MSA+MLP, MSA+MLP+CA).
- :class:`ScaleShiftTable` — the ``(num_params, dim)`` learnable table
  used by the *final* layer of many DiTs. Indexed implicitly: it
  produces scale/shift for the post-block norm.
- :func:`modulate` — ``x * (1 + scale) + shift`` with broadcasting.
- :func:`gated_residual` — ``residual + gate * branch``.

Caller composes these with ``mlx.nn.RMSNorm`` / ``LayerNorm`` and the
attention / FFN of their choice. Time embedding lives in
:mod:`mlx_arsenal.diffusion`; this module deliberately stays unaware of
it, so any conditioning source (text pooler, class label, ...) can drive
the modulation.

Example::

    from mlx_arsenal.diffusion import TimestepEmbedding
    from mlx_arsenal.modulation import (
        AdaLNModulation, modulate, gated_residual,
    )

    time = TimestepEmbedding(dim, dim)
    modulation = AdaLNModulation(dim, num_chunks=6)
    norm_msa = nn.RMSNorm(dim)

    def block_forward(x, t_emb):
        params = modulation(time(t_emb))           # (B, 6*dim)
        shift_msa, scale_msa, gate_msa, *_ = params.split(6, axis=-1)
        h = modulate(norm_msa(x), shift_msa, scale_msa)
        h = attention(h)
        return gated_residual(x, gate_msa, h)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "AdaLNModulation",
    "ScaleShiftTable",
    "gated_residual",
    "modulate",
]


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply AdaLN modulation: ``x * (1 + scale) + shift``.

    ``shift`` and ``scale`` broadcast over the sequence axis. For
    per-token modulation pass ``(B, S, dim)``; for shared modulation
    pass ``(B, dim)`` (or ``(B, 1, dim)`` for explicit broadcast).
    """
    return x * (1 + scale) + shift


def gated_residual(residual: mx.array, gate: mx.array, branch: mx.array) -> mx.array:
    """Apply a gated residual update: ``residual + gate * branch``."""
    return residual + gate * branch


class AdaLNModulation(nn.Module):
    """Project a conditioning vector to AdaLN modulation parameters.

    Implements the standard ``SiLU → Linear`` projection used by DiT,
    PixArt, ERNIE-Image, LTX-Video, and friends. The output is a flat
    tensor of shape ``(B, num_chunks * dim)`` (or ``(B, S, num_chunks * dim)``
    if conditioning is per-token); split it with ``mx.split(out,
    num_chunks, axis=-1)`` to recover the chunks.

    Args:
        dim: Model dimension. Both input and per-chunk output dim.
        num_chunks: Number of modulation chunks the conditioning
            produces. Common values:

            - ``1`` (single gate, e.g. AV cross-attention gate)
            - ``2`` (shift + scale, e.g. text cross-attention)
            - ``4`` (shift + scale, video + audio in AV models)
            - ``6`` (MSA + MLP each shift / scale / gate — vanilla DiT)
            - ``9`` (MSA + MLP + cross-attn — LTX-style)
        use_silu: Whether to pre-activate the input with SiLU before
            the Linear. Set ``False`` if the caller has already
            activated the conditioning (rare). Defaults to ``True``.
        bias: Whether the Linear includes a bias. Defaults to ``True``.

    Weight keys:
        ``linear.weight``, ``linear.bias`` (if ``bias=True``).
    """

    def __init__(
        self,
        dim: int,
        num_chunks: int,
        *,
        use_silu: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        self.dim = dim
        self.num_chunks = num_chunks
        self.use_silu = use_silu
        self.linear = nn.Linear(dim, num_chunks * dim, bias=bias)

    def __call__(self, conditioning: mx.array) -> mx.array:
        """Project ``conditioning`` to modulation params.

        Args:
            conditioning: ``(B, dim)`` or ``(B, S, dim)``.

        Returns:
            ``(B, num_chunks * dim)`` (or ``(B, S, num_chunks * dim)``).
            Caller splits along the last axis.
        """
        if self.use_silu:
            conditioning = nn.silu(conditioning)
        return self.linear(conditioning)


class ScaleShiftTable(nn.Module):
    """Final-layer scale/shift table.

    The ``(num_params, dim)`` learnable table used by the *final*
    block-level AdaLN: rather than projecting an extra conditioning
    vector, the network stores a small parameter table that is added to
    a per-batch embedded timestep before splitting into scale/shift.

    The forward signature is ``(B, dim) → (shift, scale)``. The two
    chunks are broadcast-ready for :func:`modulate`.

    Args:
        dim: Model dimension (per-chunk output dim).
        num_params: How many chunks the table stores. Typically ``2``
            (scale + shift) for the final pre-output norm.

    Weight keys:
        ``table`` — ``(num_params, dim)`` learnable parameter.
    """

    def __init__(self, dim: int, num_params: int = 2) -> None:
        super().__init__()
        self.num_params = num_params
        self.dim = dim
        # zero-init matches the reference convention (no modulation at
        # init, which keeps the pre-trained backbone's last norm output
        # close to identity in the first training steps).
        self.table = mx.zeros((num_params, dim))

    def __call__(self, embedded: mx.array) -> tuple[mx.array, ...]:
        """Combine the table with ``embedded`` and split into chunks.

        Args:
            embedded: ``(B, dim)`` — typically the post-MLP timestep
                embedding produced alongside the modulation params.

        Returns:
            Tuple of ``num_params`` arrays, each ``(B, 1, dim)``.
            Already shape-ready to broadcast over a sequence axis.
        """
        if self.table.shape[0] != self.num_params:
            raise ValueError(
                f"table must have num_params={self.num_params} rows to split "
                f"evenly, got shape {self.table.shape}"
            )
        # table is (num_params, dim); broadcast to (B, num_params, dim)
        # by adding the (B, 1, dim) embedded vector.
        combined = self.table[None, :, :] + embedded[:, None, :]
        return tuple(mx.split(combined, self.num_params, axis=1))
