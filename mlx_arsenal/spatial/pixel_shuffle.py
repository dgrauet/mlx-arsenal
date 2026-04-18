"""Pixel Shuffle / Unshuffle operations for sub-pixel convolution."""

import mlx.core as mx


def pixel_shuffle(x: mx.array, upscale_factor: int) -> mx.array:
    """Rearrange channels into spatial dimensions (sub-pixel convolution).

    Equivalent to torch.nn.functional.pixel_shuffle.

    Args:
        x: (B, H, W, C) tensor where C must be divisible by upscale_factor^2.
        upscale_factor: Factor by which to increase spatial resolution.

    Returns:
        (B, H*r, W*r, C/r^2) where r = upscale_factor.
    """
    B, H, W, C = x.shape
    r = upscale_factor
    assert C % (r * r) == 0, f"Channels {C} must be divisible by {r*r}"
    oc = C // (r * r)

    x = x.reshape(B, H, W, r, r, oc)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H, r, W, r, oc)
    return x.reshape(B, H * r, W * r, oc)


def pixel_unshuffle(x: mx.array, downscale_factor: int) -> mx.array:
    """Rearrange spatial dimensions into channels (inverse of pixel_shuffle).

    Equivalent to torch.nn.functional.pixel_unshuffle.

    Args:
        x: (B, H, W, C) tensor. H and W must be divisible by downscale_factor.
        downscale_factor: Factor by which to decrease spatial resolution.

    Returns:
        (B, H/r, W/r, C*r^2) where r = downscale_factor.
    """
    B, H, W, C = x.shape
    r = downscale_factor
    assert H % r == 0 and W % r == 0, f"H={H}, W={W} must be divisible by {r}"

    x = x.reshape(B, H // r, r, W // r, r, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H//r, W//r, r, r, C)
    return x.reshape(B, H // r, W // r, C * r * r)
