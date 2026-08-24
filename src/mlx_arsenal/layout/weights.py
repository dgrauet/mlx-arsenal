"""Weight conversion utilities for loading PyTorch models into MLX."""

from collections.abc import Callable

import mlx.core as mx

from mlx_arsenal._io import load_tensor_dict


def convert_conv_weights(weight: mx.array) -> mx.array:
    """Convert a convolution weight tensor from PyTorch to MLX format.

    PyTorch conv weights are channels-first: ``(out, in, *kernel)``. MLX
    keeps the output channel first and moves the input channel last:

    - Conv1d: ``(out, in, K)`` → ``(out, K, in)``
    - Conv2d: ``(out, in, kH, kW)`` → ``(out, kH, kW, in)``
    - Conv3d: ``(out, in, kD, kH, kW)`` → ``(out, kD, kH, kW, in)``

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
    key_map: dict[str, str] | None = None,
    key_fn: Callable[[str], str] | None = None,
    conv_keys: set[str] | None = None,
) -> dict[str, mx.array]:
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
    weights = load_tensor_dict(path)

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
