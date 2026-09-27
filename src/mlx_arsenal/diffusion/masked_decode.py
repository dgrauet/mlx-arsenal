"""Commit-step primitives for absorbing-mask diffusion LLMs (dLLMs).

Masked diffusion LLMs (LLaDA, Dream, LLaDA2.x, SDAR, Nemotron-Labs
Diffusion…) decode by repeatedly predicting every masked position and
committing ("transferring") a subset of the predictions. Every framework
(Fast-dLLM, dInfer, SGLang, LMDeploy) splits that step the same way:

1. per-position statistics of the logits (:func:`token_stats`);
2. a commit rule choosing which masked positions to fill
   (:func:`threshold_transfer`, :func:`topk_transfer`,
   :func:`factor_transfer`, :func:`entropy_bound_transfer`);
3. a per-step quota and a block schedule (:func:`transfer_schedule`,
   :func:`block_ranges`).

All functions are pure and batched per row (no batch averaging), and
compute in float32. The caller owns the model forward, the KV cache, the
mask / special token ids and the stopping logic. See
:func:`mlx_arsenal.attention.block_causal_mask` for block-causal models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx


class TokenStats(NamedTuple):
    """Per-position statistics of dLLM logits, as returned by :func:`token_stats`."""

    x0: mx.array
    """`(B, L)` int32 proposed token: argmax, or a Gumbel-max sample."""
    prob: mx.array
    """`(B, L)` float32 probability of `x0` under the temperature-1 softmax."""
    entropy: mx.array
    """`(B, L)` float32 entropy (nats) of the temperature-1 softmax."""


def token_stats(
    logits: mx.array,
    *,
    temperature: float = 0.0,
    key: mx.array | None = None,
    suppress_ids: Sequence[int] = (),
) -> TokenStats:
    """Propose a token per position with its confidence and entropy.

    `x0` is the argmax of the logits when `temperature == 0`, otherwise a
    sample of `softmax(logits / temperature)` (Gumbel-max, equivalent to the
    Fast-dLLM / LLaDA `add_gumbel_noise` + argmax). `prob` — the usual
    "low_confidence" score — and `entropy` always use the **un-noised,
    temperature-1** softmax, as in the reference implementations.

    Everything is computed in float32 (the reference code uses float64,
    which the MLX GPU lacks; near-ties may resolve differently). Entropy is
    NaN-safe for `-inf` logits.

    Args:
        logits: `(B, L, V)` logits, any float dtype.
        temperature: Sampling temperature, `>= 0`. `0` is greedy.
        key: PRNG key, required when `temperature > 0`; ignored otherwise.
        suppress_ids: Token ids never proposed (e.g. the mask token, or EOS
            before the end of the canvas). Their logits are set to `-inf`
            first, so `prob` and `entropy` are over the remaining vocabulary.

    Returns:
        :class:`TokenStats` `(x0, prob, entropy)`, each `(B, L)`.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have rank 3 (B, L, V), got shape {tuple(logits.shape)}")
    if not temperature >= 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature > 0 and key is None:
        raise ValueError("key is required when temperature > 0")
    V = logits.shape[-1]
    ids = sorted(set(suppress_ids))
    if ids and (ids[0] < 0 or ids[-1] >= V):
        raise ValueError(f"suppress_ids must lie in [0, {V}), got {ids}")
    if len(ids) >= V:
        raise ValueError("suppress_ids must leave at least one token")

    lg = logits.astype(mx.float32)
    if ids:
        suppressed = mx.any(mx.expand_dims(mx.arange(V), -1) == mx.array(ids), axis=-1)
        lg = mx.where(suppressed, float("-inf"), lg)

    if temperature > 0:
        x0 = mx.random.categorical(lg / temperature, axis=-1, key=key)
    else:
        x0 = mx.argmax(lg, axis=-1)
    x0 = x0.astype(mx.int32)

    logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    p = mx.exp(logp)
    prob = mx.take_along_axis(p, mx.expand_dims(x0, -1), axis=-1).squeeze(-1)
    entropy = -mx.sum(mx.where(p > 0, p * logp, 0.0), axis=-1)
    return TokenStats(x0=x0, prob=prob, entropy=entropy)
