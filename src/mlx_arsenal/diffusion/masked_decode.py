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
import numpy as np

from .._typing import item_int


class TokenStats(NamedTuple):
    """Per-position statistics of dLLM logits, as returned by :func:`token_stats`."""

    x0: mx.array
    """`(B, L)` int32 proposed token: argmax, or a Gumbel-max sample."""
    prob: mx.array
    """`(B, L)` float32 probability of `x0` under the temperature-1 softmax."""
    entropy: mx.array
    """`(B, L)` float32 entropy (nats) of the temperature-1 softmax."""


@mx.compile
def _gumbel_argmax(lg: mx.array, u: mx.array, inv_t: mx.array) -> mx.array:
    # argmax(l / T + Gumbel) samples softmax(l / T); u == 0 gives -inf noise.
    return mx.argmax(lg * inv_t - mx.log(-mx.log(u)), axis=-1)


@mx.compile
def _entropy(lg: mx.array, lse: mx.array) -> mx.array:
    logp = lg - lse
    p = mx.exp(logp)
    return -mx.sum(mx.where(p > 0, p * logp, 0.0), axis=-1)


def token_stats(
    logits: mx.array,
    *,
    temperature: float = 0.0,
    key: mx.array | None = None,
    suppress_ids: Sequence[int] = (),
) -> TokenStats:
    """Propose a token per position with its confidence and entropy.

    `x0` is the argmax of the logits when `temperature == 0`, otherwise a
    Gumbel-max sample of `softmax(logits / temperature)`, which is what the
    Fast-dLLM / LLaDA `add_gumbel_noise` + argmax computes. `prob` — the usual
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

    # One float32 (B, L, V) copy of the logits; the sampling noise and the
    # entropy terms are fused by mx.compile (at V ~ 262k, one such array is
    # ~1 GiB per 1024 positions). prob is gathered, not a full softmax.
    lg = logits.astype(mx.float32)
    if ids:
        bias = mx.zeros((V,), dtype=mx.float32)
        bias[mx.array(ids)] = float("-inf")
        lg = lg + bias
    lse = mx.logsumexp(lg, axis=-1, keepdims=True)

    if temperature > 0:
        u = mx.random.uniform(shape=lg.shape, key=key)
        x0 = _gumbel_argmax(lg, u, mx.array(1.0 / temperature, dtype=mx.float32))
    else:
        x0 = mx.argmax(lg, axis=-1)
    x0 = x0.astype(mx.int32)

    prob = mx.exp(mx.take_along_axis(lg, mx.expand_dims(x0, -1), axis=-1) - lse).squeeze(-1)
    entropy = _entropy(lg, lse)
    return TokenStats(x0=x0, prob=prob, entropy=entropy)


def _check_rule_inputs(scores: mx.array, candidates: mx.array, name: str) -> None:
    if scores.ndim != 2:
        raise ValueError(f"{name} must have rank 2 (B, L), got shape {tuple(scores.shape)}")
    if candidates.dtype != mx.bool_:
        raise ValueError(f"candidates must have bool dtype, got {candidates.dtype}")
    if tuple(candidates.shape) != tuple(scores.shape):
        raise ValueError(
            f"candidates shape {tuple(candidates.shape)} != {name} shape {tuple(scores.shape)}"
        )


def _rank(scores: mx.array, candidates: mx.array, *, descending: bool) -> mx.array:
    """Per-row rank of each candidate by score (0 = first); non-candidates rank last.

    Sorting is stable, so equal scores rank by position (lower first).
    Non-finite candidate scores are clamped to the finite range (NaN ranks
    last among candidates), so a non-candidate never outranks a candidate.
    """
    big = float(np.finfo(np.float32).max)
    key = scores.astype(mx.float32)
    if descending:
        key = -key
    key = mx.nan_to_num(key, nan=big, posinf=big, neginf=-big)
    key = mx.where(candidates, key, float("inf"))
    order = mx.argsort(key, axis=-1)
    return mx.argsort(order, axis=-1)


def threshold_transfer(
    confidence: mx.array,
    candidates: mx.array,
    threshold: float,
    *,
    force_one: bool = True,
) -> mx.array:
    """Commit every candidate whose confidence reaches `threshold`.

    With `force_one`, a row that has candidates but none above the threshold
    commits exactly its most confident candidate (first position on ties),
    so decoding always progresses. This is Fast-dLLM's `get_transfer_index`
    with `threshold` (and SGLang's LowConfidence); dInfer instead lowers the
    threshold to `max - 1e-5`, which can commit several near-tied tokens.

    Args:
        confidence: `(B, L)` score, higher commits first (e.g.
            :attr:`TokenStats.prob`).
        candidates: `(B, L)` bool, positions that may be committed (masked,
            and inside the current block).
        threshold: Minimum confidence, in `[0, 1]`.
        force_one: Guarantee one commit per non-empty row.

    Returns:
        `(B, L)` bool, a subset of `candidates`.
    """
    _check_rule_inputs(confidence, candidates, "confidence")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    selected = mx.logical_and(candidates, confidence.astype(mx.float32) >= threshold)
    if not force_one:
        return selected
    top1 = mx.logical_and(candidates, _rank(confidence, candidates, descending=True) == 0)
    empty = mx.logical_not(mx.any(selected, axis=-1, keepdims=True))
    return mx.logical_or(selected, mx.logical_and(empty, top1))


def topk_transfer(confidence: mx.array, candidates: mx.array, k: int | mx.array) -> mx.array:
    """Commit the `k` most confident candidates of each row.

    Pair with :func:`transfer_schedule` for LLaDA-style fixed quotas
    (`k = schedule[:, step]`). Each row uses its own `k`; a `k` above the
    row's candidate count commits all of them.

    Args:
        confidence: `(B, L)` score, higher commits first.
        candidates: `(B, L)` bool, positions that may be committed.
        k: Non-negative int, or `(B,)` integer array of per-row counts.

    Returns:
        `(B, L)` bool, a subset of `candidates`.
    """
    _check_rule_inputs(confidence, candidates, "confidence")
    B = confidence.shape[0]
    if isinstance(k, mx.array):
        if not mx.issubdtype(k.dtype, mx.integer) or tuple(k.shape) != (B,):
            raise ValueError(
                f"k must be an int or a ({B},) integer array, got {k.dtype} {tuple(k.shape)}"
            )
        if item_int(mx.min(k)) < 0:
            raise ValueError("k must be non-negative")
        kk = mx.expand_dims(k.astype(mx.int32), -1)
    else:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        kk = mx.array(k, dtype=mx.int32)
    rank = _rank(confidence, candidates, descending=True)
    return mx.logical_and(candidates, rank < kk)


def _sorted_prefix(
    scores: mx.array, candidates: mx.array, admissible: mx.array, *, descending: bool
) -> mx.array:
    """Select the longest sorted prefix whose positions are all `admissible`.

    `admissible[:, i]` refers to the `i`-th candidate in sorted order;
    positions past the row's candidate count are ignored.
    """
    count = mx.sum(candidates, axis=-1, keepdims=True)
    in_range = mx.arange(scores.shape[-1]) < count
    passed = mx.logical_and(admissible, in_range).astype(mx.int32)
    n = mx.sum(mx.cumprod(passed, axis=-1), axis=-1, keepdims=True)
    rank = _rank(scores, candidates, descending=descending)
    return mx.logical_and(candidates, rank < n)


def factor_transfer(confidence: mx.array, candidates: mx.array, factor: float) -> mx.array:
    """Commit a confidence-dependent number of candidates (Fast-dLLM factor rule).

    Candidates are sorted by confidence (descending) and the longest prefix
    of length `n` is committed such that every `i <= n` satisfies
    `c_(i) >= 1 - factor / (i + 1)`, i.e. `(i + 1)(1 - c_(i)) <= factor`.
    The most confident candidate is always committed. This is Fast-dLLM's
    `get_transfer_index_dynamic`, except for its off-by-one: when only the
    last sorted candidate fails, the reference commits every candidate; this
    function does not.

    Args:
        confidence: `(B, L)` score in `[0, 1]`, higher commits first.
        candidates: `(B, L)` bool, positions that may be committed.
        factor: Parallelism factor, `> 0`; larger commits more per step.

    Returns:
        `(B, L)` bool, a subset of `candidates` with one or more commits per
        non-empty row.
    """
    _check_rule_inputs(confidence, candidates, "confidence")
    if not factor > 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    filled = mx.where(candidates, confidence.astype(mx.float32), float("-inf"))
    sorted_conf = -mx.sort(-filled, axis=-1)
    i = mx.arange(1, confidence.shape[-1] + 1).astype(mx.float32)
    required = 1.0 - factor / (i + 1.0)
    admissible = mx.logical_or(sorted_conf >= required, i == 1)  # first always commits
    return _sorted_prefix(confidence, candidates, admissible, descending=True)


def entropy_bound_transfer(entropy: mx.array, candidates: mx.array, bound: float) -> mx.array:
    """Commit low-entropy candidates within a total entropy budget (EB-Sampler).

    Candidates are sorted by entropy (ascending) and the longest prefix is
    committed whose entropy, minus its largest term, stays within `bound`:
    `sum_{j < i} H_(j) <= bound` for every committed `i`. The lowest-entropy
    candidate is always committed. This is the EB-Sampler rule, also used by
    DiffusionGemma's `EntropyBoundSampler` to accept canvas tokens.

    Args:
        entropy: `(B, L)` non-negative per-position entropy (e.g.
            :attr:`TokenStats.entropy`), lower commits first.
        candidates: `(B, L)` bool, positions that may be committed.
        bound: Entropy budget `>= 0` (nats).

    Returns:
        `(B, L)` bool, a subset of `candidates` with one or more commits per
        non-empty row.
    """
    _check_rule_inputs(entropy, candidates, "entropy")
    if not bound >= 0:
        raise ValueError(f"bound must be >= 0, got {bound}")
    filled = mx.where(candidates, entropy.astype(mx.float32), 0.0)
    rank = _rank(entropy, candidates, descending=False)
    sorted_ent = mx.take_along_axis(filled, mx.argsort(rank, axis=-1), axis=-1)
    before = mx.cumsum(sorted_ent, axis=-1) - sorted_ent
    first = mx.arange(entropy.shape[-1]) == 0  # always commits, even with an inf entropy
    admissible = mx.logical_or(before <= bound, first)
    return _sorted_prefix(entropy, candidates, admissible, descending=False)


def transfer_schedule(num_masked: mx.array, steps: int) -> mx.array:
    """Per-step commit quota spreading each row's masked count over `steps`.

    Row `b` commits `n_b // steps` tokens per step, plus one on its first
    `n_b % steps` steps, so the quotas sum to `n_b`. This is LLaDA /
    Fast-dLLM's `get_num_transfer_tokens` (the linear-schedule expectation),
    computed per row. Use with :func:`topk_transfer`:
    `topk_transfer(conf, candidates, schedule[:, step])`.

    Args:
        num_masked: `(B,)` non-negative integer count of masked positions per
            row (typically in the current block).
        steps: Number of denoising steps, `>= 1`.

    Returns:
        `(B, steps)` int32 quotas.
    """
    if num_masked.ndim != 1:
        raise ValueError(f"num_masked must be 1D, got shape {tuple(num_masked.shape)}")
    if not mx.issubdtype(num_masked.dtype, mx.integer):
        raise ValueError(f"num_masked must have an integer dtype, got {num_masked.dtype}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if num_masked.size and item_int(mx.min(num_masked)) < 0:
        raise ValueError("num_masked must be non-negative")
    n = mx.expand_dims(num_masked.astype(mx.int32), -1)
    base = mx.floor_divide(n, steps)
    extra = mx.arange(steps) < (n - base * steps)
    return (base + extra.astype(mx.int32)).astype(mx.int32)


def block_ranges(
    prompt_len: int, gen_len: int, block_len: int, *, align: bool = False
) -> list[tuple[int, int]]:
    """Half-open `(start, end)` blocks covering the generation span.

    The span is `[prompt_len, prompt_len + gen_len)`. With `align=False`
    blocks start at `prompt_len` (LLaDA 1.x, Dream, Fast-dLLM). With
    `align=True` block boundaries sit on absolute multiples of `block_len`,
    so the first block may be partial (dInfer `start_block_align`); use this
    for block-causal models, whose blocks match
    :func:`~mlx_arsenal.attention.block_causal_mask`. The last block is
    truncated when the span is not a multiple of `block_len`.

    Args:
        prompt_len: Prompt length, `>= 0`.
        gen_len: Number of tokens to generate, `>= 1`.
        block_len: Block size, `>= 1`.
        align: Align block boundaries on absolute positions.

    Returns:
        List of `(start, end)` index pairs, in order.
    """
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be >= 0, got {prompt_len}")
    if gen_len < 1:
        raise ValueError(f"gen_len must be >= 1, got {gen_len}")
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")
    end = prompt_len + gen_len
    first = (prompt_len // block_len + 1) * block_len if align else prompt_len + block_len
    bounds = [prompt_len, *range(first, end, block_len), end]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
