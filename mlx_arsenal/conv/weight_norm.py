"""Weight normalization for MLX modules.

Implements weight normalization (Salimans & Kingma, 2016) as a wrapper
that reparameterizes a weight tensor as w = g * (v / ||v||).
"""


import mlx.core as mx
import mlx.nn as nn


class WeightNorm(nn.Module):
    """Weight normalization wrapper for any module with a weight parameter.

    Decomposes the weight into a magnitude (g) and direction (v):
        weight = g * (v / ||v||)

    Args:
        module: The module to wrap (e.g., nn.Conv1d, nn.Linear).
        weight_name: Name of the weight parameter to normalize.
        dim: Dimension over which to compute the norm. Default 0
            normalizes per output channel.
    """

    def __init__(self, module: nn.Module, weight_name: str = "weight", dim: int = 0):
        super().__init__()
        self.module = module
        self.weight_name = weight_name
        self.dim = dim

        w = getattr(module, weight_name)
        # Compute initial magnitude and direction
        axes = [i for i in range(w.ndim) if i != dim]
        self.g = mx.sqrt(mx.sum(w * w, axis=axes, keepdims=True))
        self.v = w

    def _compute_weight(self) -> mx.array:
        axes = [i for i in range(self.v.ndim) if i != self.dim]
        norm = mx.sqrt(mx.sum(self.v * self.v, axis=axes, keepdims=True) + 1e-12)
        return self.g * (self.v / norm)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        # Temporarily set the normalized weight on the module
        w = self._compute_weight()
        original = getattr(self.module, self.weight_name)
        self.module.__dict__[self.weight_name] = w
        result = self.module(x, *args, **kwargs)
        self.module.__dict__[self.weight_name] = original
        return result


def weight_norm(module: nn.Module, weight_name: str = "weight", dim: int = 0) -> WeightNorm:
    """Apply weight normalization to a module.

    Args:
        module: Module to wrap (e.g., nn.Conv1d, nn.Linear).
        weight_name: Name of the weight parameter.
        dim: Dimension for per-channel normalization.

    Returns:
        WeightNorm wrapper around the module.

    Example::

        conv = weight_norm(nn.Conv1d(16, 32, 3))
        out = conv(x)
    """
    return WeightNorm(module, weight_name, dim)
