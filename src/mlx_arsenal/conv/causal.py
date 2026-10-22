"""Causal 3D convolution with a temporal cache for chunked video processing.

Causal video VAEs (Wan 2.x, Qwen-Image, LongCat, CogVideoX, LTX) pad the
time axis on the left only, so output frame ``t`` depends on input frames
``<= t``. To encode or decode a long clip in chunks, each convolution keeps
the last few input frames of the previous chunk and prepends them to the
next one. This module implements that pattern once.
"""

from typing import Literal

import mlx.core as mx
import mlx.nn as nn


def _triple(v: int | tuple[int, int, int], name: str) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    if len(v) != 3:
        raise ValueError(f"{name} must be an int or a 3-tuple (T, H, W), got {v!r}")
    return tuple(v)


class CausalConv3d(nn.Module):
    """3D convolution that pads the time axis on the left only.

    Layout is channels-last: input ``(B, T, H, W, C_in)``, output
    ``(B, T_out, H_out, W_out, C_out)``. The weight is stored directly as
    ``weight`` with MLX layout ``(C_out, kT, kH, kW, C_in)`` (plus ``bias``),
    so PyTorch ``CausalConv3d(nn.Conv3d)`` state dicts load with only the
    usual conv-weight transpose and no key remapping.

    The temporal receptive field before the current frame is
    ``context_frames = dilation_t * (kT - 1)``. In a single call those frames
    are filled with zeros (``pad_mode="zeros"``, Wan / Qwen-Image style) or
    with copies of the first frame (``pad_mode="replicate"``, CogVideoX / LTX
    style). With :meth:`step`, they come from the previous chunk instead, and
    concatenating the chunk outputs reproduces the single-call output exactly.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: ``kT`` or ``(kT, kH, kW)``.
        stride: Stride, int or ``(sT, sH, sW)``. Chunked :meth:`step` calls
            require ``sT == 1``.
        padding: Symmetric spatial padding, int or ``(pH, pW)``. Temporal
            padding is always causal and derived from the kernel.
        dilation: Dilation, int or ``(dT, dH, dW)``.
        bias: Whether to add a learned bias.
        pad_mode: How to fill the causal context on the first chunk:
            ``"zeros"`` or ``"replicate"`` (first frame).

    A conv whose temporal padding is not ``dilation_t * (kT - 1)``, such as
    the strided ``time_conv`` of Wan's encoder downsample (``p_t = 0``, the
    caller prepends one cached frame), is a plain ``nn.Conv3d``, not this
    class.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        pad_mode: Literal["zeros", "replicate"] = "zeros",
    ):
        super().__init__()
        if pad_mode not in ("zeros", "replicate"):
            raise ValueError(f"pad_mode must be 'zeros' or 'replicate', got {pad_mode!r}")
        kernel = _triple(kernel_size, "kernel_size")
        self._stride = _triple(stride, "stride")
        self._dilation = _triple(dilation, "dilation")
        if isinstance(padding, int):
            pad_hw = (padding, padding)
        elif len(padding) == 2:
            pad_hw = tuple(padding)
        else:
            # A PyTorch-style (pT, pH, pW) would be misread as (pH, pW).
            raise ValueError(
                f"padding is spatial only, an int or (pH, pW); got {padding!r}. "
                "Temporal padding is causal and derived from the kernel."
            )
        self._padding = (0, pad_hw[0], pad_hw[1])
        self._pad_mode = pad_mode
        self.context_frames = self._dilation[0] * (kernel[0] - 1)

        scale = (in_channels * kernel[0] * kernel[1] * kernel[2]) ** -0.5
        self.weight = mx.random.uniform(-scale, scale, (out_channels, *kernel, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array, cache: mx.array | None = None) -> mx.array:
        """Convolve ``x``, optionally continuing from a previous chunk.

        Args:
            x: Input ``(B, T, H, W, C_in)``.
            cache: Up to ``context_frames`` trailing input frames of the
                previous chunk, ``(B, T_c, H, W, C_in)``. Missing context
                frames are filled according to ``pad_mode``.

        Returns:
            Output ``(B, T_out, H_out, W_out, C_out)``.
        """
        ctx = self.context_frames
        if cache is not None and ctx > 0:
            if cache.shape[1] > ctx:
                raise ValueError(
                    f"cache has {cache.shape[1]} frames but the conv uses only {ctx} context frames"
                )
            x = mx.concatenate([cache, x], axis=1)
            ctx -= cache.shape[1]
        if ctx > 0:
            if self._pad_mode == "zeros":
                x = mx.pad(x, [(0, 0), (ctx, 0), (0, 0), (0, 0), (0, 0)])
            else:
                first = mx.repeat(x[:, :1], ctx, axis=1)
                x = mx.concatenate([first, x], axis=1)
        y = mx.conv3d(
            x, self.weight, stride=self._stride, padding=self._padding, dilation=self._dilation
        )
        if "bias" in self:
            y = y + self.bias
        return y

    def step(self, x: mx.array, cache: mx.array | None = None) -> tuple[mx.array, mx.array | None]:
        """Process one chunk and return the cache for the next one.

        Calling ``step`` on consecutive chunks, threading the returned cache,
        gives the same frames as one ``__call__`` on the whole clip, for any
        chunk sizes (including 1-frame chunks).

        Args:
            x: Chunk ``(B, T, H, W, C_in)``.
            cache: Cache returned by the previous ``step`` (``None`` for the
                first chunk).

        Returns:
            ``(y, next_cache)``. ``next_cache`` holds the last
            ``context_frames`` input frames seen so far (fewer at the start of
            a clip), or ``None`` when the kernel has no temporal context.

        Raises:
            ValueError: If the temporal stride is not 1.
        """
        if self._stride[0] != 1:
            raise ValueError(f"step() needs temporal stride 1, got {self._stride[0]}")
        y = self(x, cache)
        if self.context_frames == 0:
            return y, None
        seen = x if cache is None else mx.concatenate([cache, x], axis=1)
        # A slice is a view: copy it so the cache does not pin the whole chunk.
        return y, mx.contiguous(seen[:, -self.context_frames :])
