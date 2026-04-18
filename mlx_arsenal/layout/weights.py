"""Weight conversion utilities for loading PyTorch models into MLX."""

from typing import Callable, Optional
from pathlib import Path

import mlx.core as mx


def convert_conv_weights(weight: mx.array) -> mx.array:
    """Convert a convolution weight tensor from PyTorch to MLX format.

    PyTorch conv weights: (out_ch, in_ch, *kernel_size)  [channels-first]
    MLX conv weights:     (*kernel_size, in_ch, out_ch)  [channels-last, transposed]

    Actually MLX Conv layout depends on the layer:
    - Conv1d weight: (out, kernel, in) — but loaded as (out, in, kernel) from PT
    - Conv2d weight: (out, kH, kW, in) — but loaded as (out, in, kH, kW) from PT
    - Conv3d weight: (out, kD, kH, kW, in) — but loaded as (out, in, kD, kH, kW) from PT

    This function handles the permutation for all conv dimensions.

    Args:
        weight: PyTorch-format conv weight tensor.

    Returns:
        MLX-format conv weight tensor.
    """
    ndim = weight.ndim
    if ndim == 3:
        # Conv1d: (O, I, K) -> (O, K, I)
        return weight.transpose(0, 2, 1)
    elif ndim == 4:
        # Conv2d: (O, I, kH, kW) -> (O, kH, kW, I)
        return weight.transpose(0, 2, 3, 1)
    elif ndim == 5:
        # Conv3d: (O, I, kD, kH, kW) -> (O, kD, kH, kW, I)
        return weight.transpose(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Expected 3-5D conv weight, got {ndim}D")


def load_safetensors(
    path: str,
    key_map: Optional[dict] = None,
    key_fn: Optional[Callable[[str], str]] = None,
    conv_keys: Optional[set] = None,
) -> dict:
    """Load safetensors weights with optional key remapping and conv conversion.

    Args:
        path: Path to .safetensors file.
        key_map: Optional dict mapping source keys to target keys.
            Keys not in the map are kept as-is.
        key_fn: Optional function to transform key names.
            Applied after key_map.
        conv_keys: Set of key names (after remapping) that contain
            convolution weights and should be permuted from PyTorch
            to MLX format.

    Returns:
        Dict of parameter name -> mx.array.
    """
    weights = mx.load(str(path))

    if key_map or key_fn:
        remapped = {}
        for k, v in weights.items():
            new_k = key_map.get(k, k) if key_map else k
            if key_fn:
                new_k = key_fn(new_k)
            remapped[new_k] = v
        weights = remapped

    if conv_keys:
        for k in conv_keys:
            if k in weights:
                weights[k] = convert_conv_weights(weights[k])

    return weights
