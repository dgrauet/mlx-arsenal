"""Patchify/Unpatchify operations and patch embedding layers."""

import mlx.core as mx
import mlx.nn as nn


def _as_tuple(patch_size: int | tuple[int, ...], n: int) -> tuple[int, ...]:
    """Normalize an int-or-tuple patch size to an ``n``-tuple."""
    if isinstance(patch_size, int):
        return (patch_size,) * n
    if len(patch_size) != n:
        raise ValueError(f"patch_size must be an int or a {n}-tuple, got {patch_size}")
    return tuple(patch_size)


def patchify(x: mx.array, patch_size: int | tuple[int, ...]) -> mx.array:
    """Convert spatial input into a sequence of flattened patches.

    Args:
        x: Input tensor.
            2D: (B, H, W, C) -> patches of (ph, pw)
            3D: (B, D, H, W, C) -> patches of (pd, ph, pw)
        patch_size: Patch dimensions. Int for uniform, tuple for per-dim.

    Returns:
        (B, num_patches, patch_dim) where patch_dim = prod(patch_size) * C.
    """
    if x.ndim == 4:
        # 2D: (B, H, W, C)
        B, H, W, C = x.shape
        ph, pw = _as_tuple(patch_size, 2)
        nh, nw = H // ph, W // pw
        x = x.reshape(B, nh, ph, nw, pw, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nh, nw, ph, pw, C)
        return x.reshape(B, nh * nw, ph * pw * C)

    elif x.ndim == 5:
        # 3D: (B, D, H, W, C)
        B, D, H, W, C = x.shape
        pd, ph, pw = _as_tuple(patch_size, 3)
        nd, nh, nw = D // pd, H // ph, W // pw
        x = x.reshape(B, nd, pd, nh, ph, nw, pw, C)
        x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)  # (B, nd, nh, nw, pd, ph, pw, C)
        return x.reshape(B, nd * nh * nw, pd * ph * pw * C)

    else:
        raise ValueError(f"Expected 4D or 5D input, got {x.ndim}D")


def unpatchify(
    x: mx.array,
    patch_size: int | tuple[int, ...],
    shape: tuple[int, ...],
) -> mx.array:
    """Reconstruct spatial tensor from a sequence of patches.

    Args:
        x: (B, num_patches, patch_dim) patch sequence.
        patch_size: Patch dimensions.
        shape: Target spatial shape without batch and channels.
            2D: (H, W), 3D: (D, H, W).

    Returns:
        Reconstructed tensor: (B, *shape, C).
    """
    B = x.shape[0]

    if len(shape) == 2:
        H, W = shape
        ph, pw = _as_tuple(patch_size, 2)
        nh, nw = H // ph, W // pw
        C = x.shape[-1] // (ph * pw)
        x = x.reshape(B, nh, nw, ph, pw, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nh, ph, nw, pw, C)
        return x.reshape(B, H, W, C)

    elif len(shape) == 3:
        D, H, W = shape
        pd, ph, pw = _as_tuple(patch_size, 3)
        nd, nh, nw = D // pd, H // ph, W // pw
        C = x.shape[-1] // (pd * ph * pw)
        x = x.reshape(B, nd, nh, nw, pd, ph, pw, C)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)  # (B, nd, pd, nh, ph, nw, pw, C)
        return x.reshape(B, D, H, W, C)

    else:
        raise ValueError(f"shape must have 2 or 3 elements, got {len(shape)}")


class PatchEmbed2d(nn.Module):
    """2D Patch Embedding using Conv2d.

    Projects image patches into an embedding space.

    Args:
        in_channels: Input image channels.
        embed_dim: Output embedding dimension.
        patch_size: Patch size (int or (h, w)).
        bias: Whether to use bias in the projection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        patch_size: int | tuple[int, int] = 16,
        bias: bool = True,
    ):
        super().__init__()
        ps = _as_tuple(patch_size, 2)
        self.patch_size = ps
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=ps,
            stride=ps,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (B, H, W, C) image tensor.

        Returns:
            (B, num_patches, embed_dim) patch embeddings.
        """
        x = self.proj(x)  # (B, H', W', embed_dim)
        B, H, W, D = x.shape
        return x.reshape(B, H * W, D)


class PatchEmbed3d(nn.Module):
    """3D Patch Embedding using Conv3d.

    Projects video patches into an embedding space.

    Args:
        in_channels: Input channels.
        embed_dim: Output embedding dimension.
        patch_size: Patch size (int or (d, h, w)).
        bias: Whether to use bias in the projection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        patch_size: int | tuple[int, int, int] = (2, 16, 16),
        bias: bool = True,
    ):
        super().__init__()
        ps = _as_tuple(patch_size, 3)
        self.patch_size = ps
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=ps,
            stride=ps,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (B, D, H, W, C) video tensor.

        Returns:
            (B, num_patches, embed_dim) patch embeddings.
        """
        x = self.proj(x)  # (B, D', H', W', embed_dim)
        B, D, H, W, E = x.shape
        return x.reshape(B, D * H * W, E)
