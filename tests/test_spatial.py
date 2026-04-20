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

    def test_tuple_patch_size_roundtrip(self):
        # Non-square patch via tuple — exercises the tuple-unpack branch.
        x = mx.random.normal((1, 6, 8, 3))
        patches = patchify(x, patch_size=(3, 4))
        assert patches.shape == (1, 2 * 2, 3 * 4 * 3)
        reconstructed = unpatchify(patches, patch_size=(3, 4), shape=(6, 8))
        assert mx.allclose(x, reconstructed, atol=1e-5).item()


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

    def test_int_patch_size_roundtrip(self):
        # Uniform int patch_size — exercises the int-broadcast branch.
        x = mx.random.normal((1, 4, 8, 8, 3))
        patches = patchify(x, patch_size=2)
        assert patches.shape == (1, 2 * 4 * 4, 2 * 2 * 2 * 3)
        reconstructed = unpatchify(patches, patch_size=2, shape=(4, 8, 8))
        assert mx.allclose(x, reconstructed, atol=1e-5).item()


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

    def test_int_patch_size(self):
        # Int patch_size should broadcast to (patch_size, patch_size, patch_size).
        embed = PatchEmbed3d(in_channels=3, embed_dim=64, patch_size=4)
        x = mx.random.normal((1, 8, 8, 8, 3))
        out = embed(x)
        # 8/4=2 per dim -> 2*2*2 = 8 patches
        assert out.shape == (1, 8, 64)


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


class TestPixelShufflePyTorchParity:
    """Regression tests pinning channel-packing order to torch.nn.functional.pixel_shuffle.

    The operation is defined as channel flatten order ``(out_channels, patch_row, patch_col)``.
    An earlier version of this module reversed the order to ``(patch_row, patch_col, out_channels)``;
    round-trip tests still passed, but outputs fed into downstream convs (VAE decoders, diffusion
    unpatchify) produced checkerboard artefacts. These tests compare against a literal translation
    of PyTorch's spec on channels-last tensors.
    """

    @staticmethod
    def _pt_shuffle_spec(x_nhwc, r):
        """Channels-last translation of torch.nn.functional.pixel_shuffle."""
        import numpy as np

        arr = np.array(x_nhwc).transpose(0, 3, 1, 2)
        b, c, h, w = arr.shape
        oc = c // (r * r)
        arr = arr.reshape(b, oc, r, r, h, w)
        arr = arr.transpose(0, 1, 4, 2, 5, 3)
        arr = arr.reshape(b, oc, h * r, w * r)
        return mx.array(arr.transpose(0, 2, 3, 1))

    def test_shuffle_matches_pytorch(self):
        """Channel j of input must map to output channel j // (r*r) at subpixel (j//r%r, j%r)."""
        import numpy as np

        r = 2
        B, H, W, oc = 1, 4, 4, 3
        x_np = np.zeros((B, H, W, oc * r * r), dtype=np.float32)
        for j in range(oc * r * r):
            x_np[..., j] = j
        x = mx.array(x_np)
        out = pixel_shuffle(x, upscale_factor=r)
        expected = self._pt_shuffle_spec(x, r)
        assert mx.allclose(out, expected, atol=1e-6)

    def test_unshuffle_is_true_inverse(self):
        """Unshuffle then shuffle must recover the input exactly."""
        import numpy as np

        r = 2
        x = mx.array(np.random.default_rng(0).standard_normal((1, 8, 8, 3)).astype(np.float32))
        down = pixel_unshuffle(x, downscale_factor=r)
        up = pixel_shuffle(down, upscale_factor=r)
        assert mx.allclose(x, up, atol=1e-6)

        up_via_spec = TestPixelShufflePyTorchParity._pt_shuffle_spec(down, r)
        assert mx.allclose(up, up_via_spec, atol=1e-6)
