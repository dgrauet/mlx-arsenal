"""Tests for spatial module."""

import mlx.core as mx

from mlx_arsenal.spatial import (
    PatchEmbed2d,
    PatchEmbed3d,
    patchify,
    pixel_shuffle,
    pixel_unshuffle,
    unpatchify,
    upsample_bilinear,
    upsample_nearest,
)


class TestPatchify2d:
    def test_basic(self):
        x = mx.random.normal((2, 8, 8, 3))
        patches = patchify(x, patch_size=4)
        mx.eval(patches)
        # 8/4 = 2 patches per dim, 2*2=4 patches total, 4*4*3=48 per patch
        assert patches.shape == (2, 4, 48)

    def test_roundtrip(self):
        x = mx.random.normal((1, 8, 8, 3))
        patches = patchify(x, patch_size=4)
        reconstructed = unpatchify(patches, patch_size=4, shape=(8, 8))
        mx.eval(x, reconstructed)
        assert mx.allclose(x, reconstructed, atol=1e-5)


class TestPatchify3d:
    def test_basic(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        patches = patchify(x, patch_size=(2, 4, 4))
        mx.eval(patches)
        # 4/2=2, 8/4=2, 8/4=2 -> 8 patches, 2*4*4*3=96 per patch
        assert patches.shape == (1, 8, 96)

    def test_roundtrip(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        patches = patchify(x, patch_size=(2, 4, 4))
        reconstructed = unpatchify(patches, patch_size=(2, 4, 4), shape=(4, 8, 8))
        mx.eval(x, reconstructed)
        assert mx.allclose(x, reconstructed, atol=1e-5)


class TestPatchEmbed2d:
    def test_output_shape(self):
        embed = PatchEmbed2d(in_channels=3, embed_dim=768, patch_size=16)
        x = mx.random.normal((2, 224, 224, 3))
        out = embed(x)
        mx.eval(out)
        # 224/16 = 14 patches per dim -> 196 patches
        assert out.shape == (2, 196, 768)


class TestPatchEmbed3d:
    def test_output_shape(self):
        embed = PatchEmbed3d(in_channels=3, embed_dim=512, patch_size=(2, 16, 16))
        x = mx.random.normal((1, 8, 32, 32, 3))
        out = embed(x)
        mx.eval(out)
        # 8/2=4, 32/16=2, 32/16=2 -> 16 patches
        assert out.shape == (1, 16, 512)


class TestUpsampleNearest:
    def test_2d(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = upsample_nearest(x, scale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 3)

    def test_2d_values(self):
        x = mx.arange(4).reshape(1, 2, 2, 1).astype(mx.float32)
        out = upsample_nearest(x, scale_factor=2)
        mx.eval(out)
        # Each input pixel should be replicated in a 2x2 block
        assert out[0, 0, 0, 0].item() == out[0, 0, 1, 0].item()
        assert out[0, 0, 0, 0].item() == out[0, 1, 0, 0].item()

    def test_5d(self):
        x = mx.random.normal((1, 2, 4, 4, 3))
        out = upsample_nearest(x, scale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 3)


class TestUpsampleBilinear:
    def test_shape(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = upsample_bilinear(x, scale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 3)

    def test_smooth(self):
        """Bilinear should produce smooth output."""
        x = mx.array([[[[0.0], [1.0]], [[1.0], [0.0]]]])  # (1, 2, 2, 1)
        out = upsample_bilinear(x, scale_factor=4)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 1)
        # Center values should be interpolated, not just nearest
        center = out[0, 4, 4, 0].item()
        assert 0.0 < center < 1.0


class TestPixelShuffle:
    def test_shape(self):
        x = mx.random.normal((1, 4, 4, 12))
        out = pixel_shuffle(x, upscale_factor=2)
        mx.eval(out)
        # 12 / 4 = 3 output channels, 4*2=8 spatial
        assert out.shape == (1, 8, 8, 3)

    def test_roundtrip(self):
        x = mx.random.normal((1, 4, 4, 12))
        shuffled = pixel_shuffle(x, upscale_factor=2)
        unshuffled = pixel_unshuffle(shuffled, downscale_factor=2)
        mx.eval(x, unshuffled)
        assert mx.allclose(x, unshuffled, atol=1e-5)


class TestPixelUnshuffle:
    def test_shape(self):
        x = mx.random.normal((1, 8, 8, 3))
        out = pixel_unshuffle(x, downscale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 12)
