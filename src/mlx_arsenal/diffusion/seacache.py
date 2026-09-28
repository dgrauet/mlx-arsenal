"""SeaCache: spectral-evolution-aware distance for TeaCache-style gating.

SeaCache (Chung et al., CVPR 2026) replaces TeaCache's per-model polynomial
rescale with a calibration-free spectral filter. The first-block modulated
input is passed through a Wiener-like gain built from the noise schedule and
a power-law clean-signal spectrum, and the usual accumulated relative-L1
distance is taken on the filtered tensor. Early (noisy) steps are low-passed,
so high-frequency noise no longer inflates the distance.

Combine :func:`sea_filter` with :class:`~mlx_arsenal.diffusion.TeaCacheController`
built without ``coefficients``. See ``docs/research/seacache.md``.

References:
    https://arxiv.org/abs/2602.18993
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from .._typing import item_float


def _axis_gain(
    n: int, *, half: bool, a: float, b: float, power_exp: float, eps: float
) -> tuple[mx.array, float]:
    """1-D gain on the (half or full) FFT grid of length ``n``, and its full-grid mean."""
    k = mx.arange(n, dtype=mx.float32)
    full_f = mx.minimum(k, n - k) / n  # |fftfreq(n)|
    b2 = b * b + eps

    def gain(f: mx.array) -> mx.array:
        # a·S / (a²·S + b² + eps) with S = 1 / (|f|^p + eps), rewritten without 1/eps.
        return a / (a * a + b2 * (mx.power(f, power_exp) + eps))

    full = gain(full_f)
    mean = item_float(mx.mean(full))
    if half:
        return gain(mx.arange(n // 2 + 1, dtype=mx.float32) / n), mean
    return full, mean


def sea_filter(
    x: mx.array,
    signal_scale: float,
    noise_scale: float,
    *,
    axes: Sequence[int] | None = None,
    power_exp: float = 2.0,
    eps: float = 1e-16,
) -> mx.array:
    """Apply SeaCache's SEA filter to ``x`` over its grid ``axes``.

    Each axis gets a 1-D Wiener gain from a power-law clean spectrum
    ``S(f) = 1 / (|f|^p + eps)``::

        g(f) = a·S(f) / (a²·S(f) + b² + eps)

    with ``f`` in cycles per sample. The N-D gain is the product of the
    per-axis gains, normalised to unit mean over the full frequency grid
    (SeaCache Eq. 7), and applied in the Fourier domain in float32.

    Coefficients: flow matching ``a = 1 − σ``, ``b = σ``; VP / DDPM
    ``a = √ᾱ``, ``b = √(1 − ᾱ)``. The references clamp ``σ`` to
    ``[1e-6, 1 − 1e-6]``. ``b = 0`` leaves ``x`` unchanged; ``a = 0`` returns
    zeros.

    Args:
        x: Channels-last token grid, e.g. ``(B, H, W, C)`` or
            ``(B, T, H, W, C)`` (the first-block modulated input reshaped
            to its latent grid).
        signal_scale: ``a_t``, the clean-signal coefficient (``>= 0``).
        noise_scale: ``b_t``, the noise coefficient (``>= 0``).
        axes: Axes to filter. Default: every axis except the first
            (batch) and the last (channels).
        power_exp: Exponent ``p`` of the clean spectrum. SeaCache uses 2
            for images and 3 for video.
        eps: Regulariser of the spectrum and of the gain (``> 0``).

    Returns:
        The filtered tensor, same shape and dtype as ``x``.
    """
    if signal_scale < 0 or noise_scale < 0:
        raise ValueError(
            f"signal_scale and noise_scale must be >= 0, got {signal_scale}, {noise_scale}"
        )
    if power_exp <= 0:
        raise ValueError(f"power_exp must be > 0, got {power_exp}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if axes is None:
        if x.ndim < 3:
            raise ValueError(
                f"default axes need a (B, ..., C) grid with ndim >= 3, got shape {x.shape}"
            )
        axes = range(1, x.ndim - 1)
    norm: list[int] = []
    for ax in axes:
        if not -x.ndim <= ax < x.ndim:
            raise ValueError(f"axis {ax} out of range for ndim {x.ndim}")
        norm.append(ax % x.ndim)
    if not norm or len(set(norm)) != len(norm):
        raise ValueError(f"axes must be non-empty and unique, got {tuple(axes)}")

    gain = mx.array(1.0, dtype=mx.float32)
    mean = 1.0
    for i, ax in enumerate(norm):
        g, m = _axis_gain(
            x.shape[ax],
            half=i == len(norm) - 1,
            a=signal_scale,
            b=noise_scale,
            power_exp=power_exp,
            eps=eps,
        )
        shape = [1] * x.ndim
        shape[ax] = g.shape[0]
        g = g.reshape(shape)
        gain = gain * g
        mean *= m  # the mean of a separable product is the product of the means
    if mean > 0:
        gain = gain / mean

    sizes = tuple(x.shape[ax] for ax in norm)
    spectrum = mx.fft.rfftn(x.astype(mx.float32), s=sizes, axes=norm)
    out = mx.fft.irfftn(spectrum * gain, s=sizes, axes=norm)
    return out.astype(x.dtype)
