"""Uniform-noise diffusion LLM decoding helpers (DiffusionGemma).

Unlike absorbing-mask dLLMs (see :mod:`mlx_arsenal.diffusion.masked_decode`),
uniform-noise dLLMs such as DiffusionGemma start from a canvas of random
tokens and re-decide the whole canvas at every step: the lowest-entropy
predictions are accepted (the EB-Sampler rule,
:func:`~mlx_arsenal.diffusion.entropy_bound_transfer` over all positions)
and every other position is re-noised with fresh uniform tokens. The final
output is the argmax canvas of the last step. Temperature follows a linear
schedule, and decoding stops early once the argmax canvas is stable and
confident.

This module ships the pieces that are not already in ``masked_decode``:
:func:`linear_temperature`, :func:`uniform_canvas`, :func:`renoise` and
:class:`StableConfidentStopping`. Sampling from ``softmax(logits / t)`` is a
single ``mx.random.categorical`` call on the caller side; self-conditioning
and the encoder cache are model-specific. Semantics follow Hugging Face's
``generation_diffusion_gemma.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx


def linear_temperature(
    remaining: int, num_steps: int, *, t_min: float = 0.4, t_max: float = 0.8
) -> float:
    """Temperature of a linear schedule indexed by the number of remaining steps.

    ``t = t_min + (t_max - t_min) * remaining / num_steps``: the first step
    (``remaining == num_steps``) uses ``t_max`` and the schedule decays
    towards ``t_min``. Diffusion loops count steps down, so ``remaining``
    runs ``num_steps, ..., 1`` (Hugging Face's
    ``LinearTemperatureScheduleLogitsProcessor``; DiffusionGemma uses
    ``t_min = 0.4``, ``t_max = 0.8``, 48 steps).

    Args:
        remaining: Steps remaining, in ``[0, num_steps]``.
        num_steps: Maximum number of denoising steps, ``>= 1``.
        t_min: Final temperature, ``> 0``.
        t_max: Initial temperature, ``> 0`` (usually ``>= t_min``; a rising
            schedule is accepted, as in the reference).

    Returns:
        The temperature to divide the logits by.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if not 0 <= remaining <= num_steps:
        raise ValueError(f"remaining must be in [0, {num_steps}], got {remaining}")
    if not t_min > 0 or not t_max > 0:
        raise ValueError(f"t_min and t_max must be > 0, got t_min={t_min}, t_max={t_max}")
    return t_min + (t_max - t_min) * remaining / num_steps


def uniform_canvas(
    shape: Sequence[int], vocab_size: int, *, key: mx.array | None = None
) -> mx.array:
    """Uniformly random tokens, for the initial canvas and for re-noising.

    With ``key=None`` the global MLX PRNG is used, exactly like
    ``mx.random.randint(0, vocab_size, shape)``, so a loop that seeds the
    global PRNG reproduces a reference implementation's draws call for call.

    Args:
        shape: Canvas shape, e.g. ``(B, L)``, positive sizes.
        vocab_size: Number of token ids, ``>= 1``.
        key: Optional PRNG key.

    Returns:
        int32 tokens in ``[0, vocab_size)``.
    """
    dims = tuple(shape)
    if not dims or any(d < 1 for d in dims):
        raise ValueError(f"shape must be non-empty with positive sizes, got {dims}")
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    return mx.random.randint(0, vocab_size, dims, key=key).astype(mx.int32)


def renoise(canvas: mx.array, accepted: mx.array, noise: mx.array) -> mx.array:
    """Keep accepted tokens and replace every other position with noise.

    ``where(accepted, canvas, noise)``, with ``noise`` typically a fresh
    :func:`uniform_canvas`. Passing the noise in keeps this function pure and
    lets a caller reproduce a reference's random draws.

    Args:
        canvas: Integer tokens (the accepted canvas).
        accepted: Bool mask, same shape, True where the token is kept.
        noise: Integer replacement tokens, same shape.

    Returns:
        Tokens with ``canvas``'s shape and dtype.
    """
    if tuple(accepted.shape) != tuple(canvas.shape) or tuple(noise.shape) != tuple(canvas.shape):
        raise ValueError(
            f"canvas, accepted and noise must share one shape, got {tuple(canvas.shape)}, "
            f"{tuple(accepted.shape)}, {tuple(noise.shape)}"
        )
    if accepted.dtype != mx.bool_:
        raise ValueError(f"accepted must have bool dtype, got {accepted.dtype}")
    if not mx.issubdtype(canvas.dtype, mx.integer) or not mx.issubdtype(noise.dtype, mx.integer):
        raise ValueError(f"canvas and noise must be integer, got {canvas.dtype} and {noise.dtype}")
    return mx.where(accepted, canvas, noise.astype(canvas.dtype))


class StableConfidentStopping:
    """Stop decoding a canvas once its argmax is stable and confident.

    Per batch row, stop when the argmax canvas equals each of the previous
    ``stability_threshold`` argmax canvases (``0`` makes every call stable)
    and the mean per-position entropy is below ``confidence_threshold``.
    Before ``stability_threshold`` canvases have been seen, no row is stable.
    This is Hugging Face's ``StableAndConfidentStoppingCriteria``
    (DiffusionGemma defaults: 1 and 0.005); pass the entropy of the
    temperature-scaled logits, as the reference does.

    Call once per denoising step; :meth:`reset` before each new canvas.

    Args:
        stability_threshold: Number of previous identical canvases, ``>= 0``.
        confidence_threshold: Mean-entropy bound (nats), ``> 0``.
    """

    def __init__(self, stability_threshold: int = 1, confidence_threshold: float = 0.005):
        if stability_threshold < 0:
            raise ValueError(f"stability_threshold must be >= 0, got {stability_threshold}")
        if not confidence_threshold > 0:
            raise ValueError(f"confidence_threshold must be > 0, got {confidence_threshold}")
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.reset()

    def reset(self) -> None:
        """Forget previous canvases. Call before decoding a new canvas."""
        self._history: list[mx.array] = []

    def __call__(self, argmax_canvas: mx.array, entropy: mx.array) -> mx.array:
        """Decide, per row, whether to stop after this step.

        Args:
            argmax_canvas: ``(B, L)`` argmax tokens of this step.
            entropy: ``(B, L)`` per-position entropy of this step.

        Returns:
            ``(B,)`` bool.
        """
        if argmax_canvas.ndim != 2 or tuple(entropy.shape) != tuple(argmax_canvas.shape):
            raise ValueError(
                f"argmax_canvas and entropy must share a (B, L) shape, got "
                f"{tuple(argmax_canvas.shape)} and {tuple(entropy.shape)}"
            )
        if self._history and tuple(self._history[-1].shape) != tuple(argmax_canvas.shape):
            raise ValueError(
                f"canvas shape changed from {tuple(self._history[-1].shape)} to "
                f"{tuple(argmax_canvas.shape)}; call reset() for a new canvas"
            )
        B = argmax_canvas.shape[0]
        if self.stability_threshold == 0:
            stable = mx.ones((B,), dtype=mx.bool_)
        elif len(self._history) < self.stability_threshold:
            stable = mx.zeros((B,), dtype=mx.bool_)
        else:
            stable = mx.ones((B,), dtype=mx.bool_)
            for previous in self._history:
                stable = mx.logical_and(stable, mx.all(mx.equal(previous, argmax_canvas), axis=-1))
        if self.stability_threshold > 0:
            # Materialize the small (B, L) canvas: kept lazy, it would pin the
            # (B, L, V) logits it was computed from until it leaves the history.
            mx.eval(argmax_canvas)
            self._history.append(argmax_canvas)
            self._history = self._history[-self.stability_threshold :]
        confident = mx.mean(entropy.astype(mx.float32), axis=-1) < self.confidence_threshold
        return mx.logical_and(stable, confident)
