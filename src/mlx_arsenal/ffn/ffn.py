"""Feed-forward / MLP blocks used by transformer architectures.

Two flavours cover the bulk of real-world ports:

- :class:`FeedForward` — the classic 2-Linear MLP
  ``Linear(d, d_inner) → activation → Linear(d_inner, d_out)``.
  Configurable activation (``gelu``, ``gelu_approx``, ``silu``,
  ``relu``). Used by LTX-Video, vanilla DiT, T5, most early
  transformers.

- :class:`GatedFFN` — the gated 3-Linear MLP
  ``down(act(gate(x)) * up(x))``. ``gate_activation`` selects the
  variant: ``"gelu"`` is GeGLU (Shazeer 2020), ``"silu"`` is SwiGLU
  (LLaMA / PaLM family). Used by ERNIE-Image, modern LLM-style
  transformers, mixture-of-experts experts.

Weight-key conventions follow the most common public naming:

- :class:`FeedForward` → ``proj_in.{weight,bias}``,
  ``proj_out.{weight,bias}`` (LTX / vanilla naming).
- :class:`GatedFFN` → ``gate_proj.weight``, ``up_proj.weight``,
  ``down_proj.weight`` (LLaMA / HF naming).

Ports that load from diffusers-style ``net.0/net.2`` or Megatron-style
``linear_fc1/linear_fc2`` should remap keys at load time
(``mlx_arsenal.layout.load_safetensors(..., key_map=...)``); arsenal's
job is to provide the math, not chase every upstream's bikeshed.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

__all__ = ["FeedForward", "GatedFFN", "GeGLU", "SwiGLU"]


_ACTIVATIONS: dict[str, Callable[[mx.array], mx.array]] = {
    "gelu": nn.gelu,
    "gelu_approx": nn.gelu_approx,
    "silu": nn.silu,
    "relu": nn.relu,
}


def _resolve_activation(
    activation: str | Callable[[mx.array], mx.array],
) -> Callable[[mx.array], mx.array]:
    if not isinstance(activation, str):
        return activation
    try:
        return _ACTIVATIONS[activation]
    except KeyError as exc:
        valid = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"unknown activation {activation!r}; valid: {valid}") from exc


def _resolve_inner_dim(dim: int, inner_dim: int | None, mult: float) -> int:
    if inner_dim is not None:
        return inner_dim
    return int(dim * mult)


class FeedForward(nn.Module):
    """Classic 2-Linear feed-forward block.

    Args:
        dim: Input dimension.
        dim_out: Output dimension. Defaults to ``dim`` (square FFN).
        mult: Multiplier on ``dim`` to derive the inner dimension when
            ``inner_dim`` is not supplied. Defaults to ``4.0``.
        inner_dim: Explicit inner dimension. Overrides ``mult``.
        activation: Either a name (``"gelu"``, ``"gelu_approx"``,
            ``"silu"``, ``"relu"``) or a callable. Defaults to
            ``"gelu"``.
        bias: Whether the two Linears include bias. Defaults to ``True``.

    Weight keys:
        ``proj_in.weight``, ``proj_in.bias`` (if ``bias=True``),
        ``proj_out.weight``, ``proj_out.bias`` (if ``bias=True``).
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4.0,
        *,
        inner_dim: int | None = None,
        activation: str | Callable[[mx.array], mx.array] = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        inner = _resolve_inner_dim(dim, inner_dim, mult)
        self._activation = _resolve_activation(activation)
        self.proj_in = nn.Linear(dim, inner, bias=bias)
        self.proj_out = nn.Linear(inner, dim_out, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(self._activation(self.proj_in(x)))


class GatedFFN(nn.Module):
    """Gated 3-Linear feed-forward block (GeGLU / SwiGLU).

    Computes ``down_proj( activation(gate_proj(x)) * up_proj(x) )``.

    Args:
        dim: Input dimension.
        dim_out: Output dimension. Defaults to ``dim``.
        mult: Multiplier on ``dim`` for the inner dimension when
            ``inner_dim`` is not supplied. Defaults to ``4.0``. (Many
            LLaMA-family models use ``mult ≈ 2.67`` to keep param count
            comparable to a 4× standard FFN despite the extra
            projection — caller supplies ``inner_dim`` directly in
            that case.)
        inner_dim: Explicit inner dimension. Overrides ``mult``.
        gate_activation: ``"gelu"`` for GeGLU (Shazeer 2020), ``"silu"``
            for SwiGLU (LLaMA / PaLM). Also accepts ``"gelu_approx"``
            or a callable.
        bias: Whether the three Linears include bias. Defaults to
            ``False`` (matches LLaMA / ERNIE).

    Weight keys:
        ``gate_proj.weight``, ``up_proj.weight``, ``down_proj.weight``
        (and corresponding ``.bias`` keys if ``bias=True``).
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4.0,
        *,
        inner_dim: int | None = None,
        gate_activation: str | Callable[[mx.array], mx.array] = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        inner = _resolve_inner_dim(dim, inner_dim, mult)
        self._activation = _resolve_activation(gate_activation)
        self.gate_proj = nn.Linear(dim, inner, bias=bias)
        self.up_proj = nn.Linear(dim, inner, bias=bias)
        self.down_proj = nn.Linear(inner, dim_out, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self._activation(self.gate_proj(x)) * self.up_proj(x))


def GeGLU(  # noqa: N802 — convention name
    dim: int,
    dim_out: int | None = None,
    mult: float = 4.0,
    *,
    inner_dim: int | None = None,
    bias: bool = False,
) -> GatedFFN:
    """:class:`GatedFFN` with ``gate_activation="gelu"`` (Shazeer 2020)."""
    return GatedFFN(
        dim,
        dim_out=dim_out,
        mult=mult,
        inner_dim=inner_dim,
        gate_activation="gelu",
        bias=bias,
    )


def SwiGLU(  # noqa: N802 — convention name
    dim: int,
    dim_out: int | None = None,
    mult: float = 4.0,
    *,
    inner_dim: int | None = None,
    bias: bool = False,
) -> GatedFFN:
    """:class:`GatedFFN` with ``gate_activation="silu"`` (LLaMA / PaLM)."""
    return GatedFFN(
        dim,
        dim_out=dim_out,
        mult=mult,
        inner_dim=inner_dim,
        gate_activation="silu",
        bias=bias,
    )
