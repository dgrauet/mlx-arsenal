"""Tests for tiling module."""

from typing import cast

import mlx.core as mx
import pytest

from mlx_arsenal.tiling import temporal_slice_process, tiled_process
from mlx_arsenal.tiling._blend import blend_weight_1d, window_starts


class TestTiledProcess:
    def test_small_input_passthrough(self):
        """Input smaller than tile_size should be processed directly."""
        x = mx.random.normal((1, 4, 4, 3))
        out = tiled_process(x, fn=lambda t: t * 2, tile_size=8, overlap=2)
        mx.eval(out)
        expected = x * 2
        mx.eval(expected)
        assert mx.allclose(out, expected, atol=1e-5).item()

    def test_tiled_shape(self):
        """Output should match the shape of fn(full_input)."""
        x = mx.random.normal((1, 16, 16, 3))
        out = tiled_process(x, fn=lambda t: t, tile_size=8, overlap=2)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 3)

    def test_identity_function(self):
        """Tiled identity should approximately reproduce the input."""
        x = mx.random.normal((1, 16, 16, 3))
        out = tiled_process(x, fn=lambda t: t, tile_size=8, overlap=4)
        mx.eval(out)
        # Should be very close to input since we're applying identity
        assert mx.allclose(out, x, atol=1e-4).item()


class TestTemporalSliceProcess:
    def test_small_input_passthrough(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        out = temporal_slice_process(x, fn=lambda t: t * 2, window_size=8)
        mx.eval(out)
        expected = x * 2
        mx.eval(expected)
        assert mx.allclose(out, expected, atol=1e-5).item()

    def test_temporal_shape(self):
        x = mx.random.normal((1, 32, 4, 4, 3))
        out = temporal_slice_process(x, fn=lambda t: t, window_size=8, overlap=2)
        mx.eval(out)
        assert out.shape == (1, 32, 4, 4, 3)

    def test_identity_function(self):
        x = mx.random.normal((1, 16, 4, 4, 3))
        out = temporal_slice_process(x, fn=lambda t: t, window_size=8, overlap=4)
        mx.eval(out)
        assert mx.allclose(out, x, atol=1e-4).item()

    def test_unaligned_length_appends_final_window(self):
        # T=30, window_size=8, overlap=2 -> stride=6. Iter starts = [0,6,12,18].
        # Last covers 18..26 < 30, so the final-window append path runs.
        x = mx.random.normal((1, 30, 4, 4, 3))
        out = temporal_slice_process(x, fn=lambda t: t, window_size=8, overlap=2)
        assert out.shape == (1, 30, 4, 4, 3)
        assert mx.allclose(out, x, atol=1e-4).item()


def test_tiled_process_non_square_axis_smaller_than_tile():
    # One spatial axis fits inside a single tile, the other needs tiling.
    x = mx.ones((1, 32, 8, 3))
    out = tiled_process(x, lambda t: t, tile_size=16, overlap=4)
    assert out.shape == x.shape
    assert mx.allclose(out, x, atol=1e-5).item()


def test_temporal_overlap_ge_window_raises():
    x = mx.ones((1, 32, 4, 4, 3))
    with pytest.raises(ValueError):
        temporal_slice_process(x, lambda t: t, window_size=8, overlap=8)
    with pytest.raises(ValueError):
        temporal_slice_process(x, lambda t: t, window_size=8, overlap=9)


class TestTiledProcessNonIdentity:
    def test_scale_fn_exact_in_blend_zones(self):
        """fn = x*2 must give exactly 2x everywhere, including blend zones."""
        x = mx.random.normal((1, 16, 16, 3))
        out = tiled_process(x, fn=lambda t: t * 2, tile_size=8, overlap=4)
        assert mx.allclose(out, x * 2, atol=1e-4).item()

    def test_downscaling_fn_matches_full_image_pool(self):
        """2x2 average pool (VAE-encoder-like) through tiling matches the
        full-image pooled result: tile starts (stride=4) are even, so every
        tile's pooled output is an exact slice of the full pooled image."""

        def pool2(t: mx.array) -> mx.array:
            B, H, W, C = t.shape
            return mx.mean(t.reshape(B, H // 2, 2, W // 2, 2, C), axis=(2, 4))

        x = mx.random.normal((1, 16, 16, 2))
        out = tiled_process(x, fn=pool2, tile_size=8, overlap=4)
        expected = pool2(x)
        assert out.shape == (1, 8, 8, 2)
        assert mx.allclose(out, expected, atol=1e-4).item()

    def test_spatial_dims_override_channels_first(self):
        """(B, C, H, W)-style layout with spatial_dims=(2, 3)."""
        x = mx.random.normal((1, 3, 16, 16))
        out = tiled_process(x, fn=lambda t: t, tile_size=8, overlap=4, spatial_dims=(2, 3))
        assert out.shape == (1, 3, 16, 16)
        assert mx.allclose(out, x, atol=1e-4).item()

    def test_overlap_ge_tile_size_raises(self):
        x = mx.ones((1, 16, 16, 3))
        with pytest.raises(ValueError, match="overlap"):
            tiled_process(x, lambda t: t, tile_size=8, overlap=8)
        with pytest.raises(ValueError, match="overlap"):
            tiled_process(x, lambda t: t, tile_size=8, overlap=9)


class TestTemporalSliceProcessVariants:
    def test_temporal_dim_override(self):
        """Temporal axis at dim 2 instead of the default dim 1."""
        x = mx.random.normal((1, 3, 32, 4))
        out = temporal_slice_process(x, fn=lambda t: t, window_size=8, overlap=2, temporal_dim=2)
        assert out.shape == (1, 3, 32, 4)
        assert mx.allclose(out, x, atol=1e-4).item()

    def test_non_identity_fn_on_long_input(self):
        """fn = x*2 through the windowed path (T > window_size)."""
        x = mx.random.normal((1, 32, 4, 4, 3))
        out = temporal_slice_process(x, fn=lambda t: t * 2, window_size=8, overlap=4)
        assert mx.allclose(out, x * 2, atol=1e-4).item()


class TestBlendWeight1d:
    def test_no_blend_is_all_ones(self):
        w = blend_weight_1d(8, blend_left=0, blend_right=0)
        assert mx.array_equal(w, mx.ones((8,))).item()

    def test_left_ramp_strictly_increasing_middle_one(self):
        w = blend_weight_1d(8, blend_left=3, blend_right=0)
        vals = cast(list[float], w.tolist())
        # Ramp excludes the endpoints 0 and 1: strictly increasing, < 1.
        assert 0.0 < vals[0] < vals[1] < vals[2] < 1.0
        # Middle is exactly 1.0.
        assert vals[3:] == [1.0] * 5

    def test_right_ramp_strictly_decreasing_middle_one(self):
        w = blend_weight_1d(8, blend_left=0, blend_right=3)
        vals = cast(list[float], w.tolist())
        assert vals[:5] == [1.0] * 5
        assert 1.0 > vals[5] > vals[6] > vals[7] > 0.0

    def test_both_ramps(self):
        w = blend_weight_1d(9, blend_left=2, blend_right=2)
        vals = cast(list[float], w.tolist())
        assert vals[0] < vals[1] < 1.0
        assert vals[2:7] == [1.0] * 5
        assert 1.0 > vals[7] > vals[8] > 0.0


class TestWindowStarts:
    def test_aligned_grid(self):
        assert window_starts(16, 8, 4) == [0, 4, 8]

    def test_final_window_appended_when_needed(self):
        # Strided starts [0, 6, 12, 18] end at 26 < 30 -> 22 appended.
        assert window_starts(30, 8, 6) == [0, 6, 12, 18, 22]

    def test_single_window(self):
        assert window_starts(8, 8, 4) == [0]

    def test_covers_total(self):
        for total, window, stride in [(16, 8, 4), (30, 8, 6), (17, 8, 5), (9, 8, 8)]:
            starts = window_starts(total, window, stride)
            covered = set()
            for s in starts:
                assert 0 <= s and s + window <= total
                covered.update(range(s, s + window))
            assert covered == set(range(total))
