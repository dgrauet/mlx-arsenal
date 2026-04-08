"""Scaled Dot-Product Attention helpers.

Wraps mx.fast.scaled_dot_product_attention with common boilerplate:
reshaping between (B, L, H*D) and (B, H, L, D), GQA/MQA key-value
repetition, and optional mask handling.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """Repeat KV heads for Grouped Query Attention.

    Args:
        x: (B, n_kv_heads, L, D)
        n_rep: Number of times to repeat each KV head.

    Returns:
        (B, n_kv_heads * n_rep, L, D)
    """
    if n_rep == 1:
        return x
    B, n_kv, L, D = x.shape
    x = mx.expand_dims(x, axis=2)  # (B, n_kv, 1, L, D)
    x = mx.repeat(x, n_rep, axis=2)  # (B, n_kv, n_rep, L, D)
    return x.reshape(B, n_kv * n_rep, L, D)


def sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array] = None,
    scale: Optional[float] = None,
    n_heads: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
) -> mx.array:
    """Scaled dot-product attention with automatic reshape and GQA support.

    Handles the common pattern of:
    1. Reshaping from (B, L, H*D) to (B, H, L, D)
    2. Repeating KV heads for GQA/MQA
    3. Calling SDPA
    4. Reshaping back to (B, L, H*D)

    If inputs are already 4D (B, H, L, D), skips reshaping.

    Args:
        q: Query tensor. (B, L, H*D) or (B, H, L, D).
        k: Key tensor. Same format as q.
        v: Value tensor. Same format as q.
        mask: Optional attention mask.
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim).
        n_heads: Number of query heads. Required if input is 3D.
        n_kv_heads: Number of KV heads. If different from n_heads, KV
            heads are repeated for GQA.

    Returns:
        Attention output, same format as input.
    """
    needs_reshape = q.ndim == 3

    if needs_reshape:
        assert n_heads is not None, "n_heads required for 3D input"
        B, L, _ = q.shape
        D = q.shape[-1] // n_heads

        q = q.reshape(B, L, n_heads, D).transpose(0, 2, 1, 3)

        if n_kv_heads is None:
            n_kv_heads = n_heads
        k = k.reshape(B, -1, n_kv_heads, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, n_kv_heads, D).transpose(0, 2, 1, 3)

    if n_kv_heads is not None and n_heads is not None and n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    if needs_reshape:
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

    return out


class SDPAttention(nn.Module):
    """Multi-head attention module using scaled dot-product attention.

    A thin wrapper that manages Q/K/V projections and calls sdpa().

    Args:
        dim: Model dimension.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of KV heads (for GQA). Defaults to n_heads.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        kv: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: (B, L, D) input tensor.
            mask: Optional attention mask.
            kv: Optional cross-attention source. If None, self-attention.

        Returns:
            (B, L, D) output tensor.
        """
        q = self.q_proj(x)
        source = kv if kv is not None else x
        k = self.k_proj(source)
        v = self.v_proj(source)

        out = sdpa(
            q, k, v,
            mask=mask,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
        )
        return self.o_proj(out)
