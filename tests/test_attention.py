"""Tests for attention module (masks)."""

import math
from typing import cast

import mlx.core as mx
import pytest

from mlx_arsenal.attention import (
    causal_mask,
    radial_box_mask,
    radial_gaussian_mask,
    sliding_tile_block_mask,
    sliding_tile_centered_mask,
    sliding_window_mask,
    spatial_only_mask,
    temporal_only_mask,
)


class TestCausalMask:
    def test_shape(self):
        m = causal_mask(seq_len=5)
        assert m.shape == (1, 1, 5, 5)

    def test_lower_triangular(self):
        m = cast(list[list[float]], causal_mask(seq_len=4)[0, 0].tolist())
        for i in range(4):
            for j in range(4):
                value = m[i][j]
                if j <= i:
                    assert value == 0.0
                else:
                    assert math.isinf(value) and value < 0

    def test_kv_offset(self):
        m = causal_mask(seq_len=2, offset=3)
        assert m.shape == (1, 1, 2, 5)
        grid = cast(list[list[float]], m[0, 0].tolist())
        # Row 0 (absolute pos 3) can see cols 0..3.
        assert grid[0][3] == 0.0
        assert math.isinf(grid[0][4])
        # Row 1 (absolute pos 4) can see cols 0..4.
        assert grid[1][4] == 0.0

    def test_dtype(self):
        m = causal_mask(seq_len=3, dtype=mx.float16)
        assert m.dtype == mx.float16


class TestSlidingWindowMask:
    def test_shape(self):
        m = sliding_window_mask(seq_len=6, window_size=3)
        assert m.shape == (1, 1, 6, 6)

    def test_window_limits_attention(self):
        m = cast(list[list[float]], sliding_window_mask(seq_len=5, window_size=2)[0, 0].tolist())
        # window_size=2 means each position attends to itself + 1 prior.
        for i in range(5):
            for j in range(5):
                value = m[i][j]
                if i - 1 <= j <= i:
                    assert value == 0.0
                else:
                    assert math.isinf(value) and value < 0

    def test_window_larger_than_seq_is_fully_causal(self):
        causal = causal_mask(seq_len=4)[0, 0]
        windowed = sliding_window_mask(seq_len=4, window_size=100)[0, 0]
        assert mx.array_equal(causal, windowed).item()


class TestSpatialOnlyMask:
    def test_shape(self):
        m = spatial_only_mask(T=2, H=3, W=4)
        assert m.shape == (1, 1, 24, 24)

    def test_pattern(self):
        # T=2, H=2, W=2 → S=8. Frame 0 = tokens 0..3, frame 1 = tokens 4..7.
        m = cast(list[list[float]], spatial_only_mask(T=2, H=2, W=2)[0, 0].tolist())
        for i in range(8):
            for j in range(8):
                same_frame = (i // 4) == (j // 4)
                if same_frame:
                    assert m[i][j] == 0.0
                else:
                    assert math.isinf(m[i][j]) and m[i][j] < 0

    def test_dtype(self):
        m = spatial_only_mask(T=2, H=2, W=2, dtype=mx.float16)
        assert m.dtype == mx.float16

    def test_validation(self):
        with pytest.raises(ValueError):
            spatial_only_mask(T=0, H=2, W=2)
        with pytest.raises(ValueError):
            spatial_only_mask(T=2, H=-1, W=2)
        with pytest.raises(ValueError):
            spatial_only_mask(T=2, H=2, W=0)


class TestTemporalOnlyMask:
    def test_shape(self):
        m = temporal_only_mask(T=2, H=3, W=4)
        assert m.shape == (1, 1, 24, 24)

    def test_pattern(self):
        # T=2, H=2, W=2. Token index i = t*4 + h*2 + w. Same (h,w) ⇔ i%4 == j%4.
        m = cast(list[list[float]], temporal_only_mask(T=2, H=2, W=2)[0, 0].tolist())
        for i in range(8):
            for j in range(8):
                same_pos = (i % 4) == (j % 4)
                if same_pos:
                    assert m[i][j] == 0.0
                else:
                    assert math.isinf(m[i][j]) and m[i][j] < 0

    def test_dtype(self):
        m = temporal_only_mask(T=2, H=2, W=2, dtype=mx.float16)
        assert m.dtype == mx.float16

    def test_validation(self):
        with pytest.raises(ValueError):
            temporal_only_mask(T=0, H=2, W=2)
        with pytest.raises(ValueError):
            temporal_only_mask(T=2, H=2, W=-1)


class TestSlidingTileCenteredMask:
    def test_shape(self):
        m = sliding_tile_centered_mask(T=2, H=3, W=4, window=(0, 1, 1))
        assert m.shape == (1, 1, 24, 24)

    def test_pattern_zero_window(self):
        # window=(0,0,0) → only attend to self.
        mask = sliding_tile_centered_mask(T=2, H=2, W=2, window=(0, 0, 0))[0, 0]
        m = cast(list[list[float]], mask.tolist())
        for i in range(8):
            for j in range(8):
                if i == j:
                    assert m[i][j] == 0.0
                else:
                    assert math.isinf(m[i][j]) and m[i][j] < 0

    def test_pattern_spatial_only_window(self):
        # window=(0, large, large) ≡ spatial_only.
        m = sliding_tile_centered_mask(T=2, H=2, W=2, window=(0, 99, 99))[0, 0]
        ref = spatial_only_mask(T=2, H=2, W=2)[0, 0]
        assert mx.array_equal(m, ref).item()

    def test_pattern_temporal_only_window(self):
        # window=(large, 0, 0) ≡ temporal_only.
        m = sliding_tile_centered_mask(T=2, H=2, W=2, window=(99, 0, 0))[0, 0]
        ref = temporal_only_mask(T=2, H=2, W=2)[0, 0]
        assert mx.array_equal(m, ref).item()

    def test_validation(self):
        with pytest.raises(ValueError):
            sliding_tile_centered_mask(T=2, H=2, W=2, window=(-1, 0, 0))
        with pytest.raises(ValueError):
            sliding_tile_centered_mask(T=0, H=2, W=2, window=(1, 1, 1))


class TestSlidingTileBlockMask:
    def test_shape(self):
        m = sliding_tile_block_mask(T=2, H=4, W=4, tile=(1, 2, 2))
        assert m.shape == (1, 1, 32, 32)

    def test_block_alignment(self):
        # All queries in the same tile must have identical mask rows.
        T, H, W = 2, 4, 4
        tile = (1, 2, 2)
        m = sliding_tile_block_mask(T=T, H=H, W=W, tile=tile, window=(0, 1, 1))[0, 0]
        S = T * H * W
        rows = cast(list[list[float]], m.tolist())
        tile_ids = []
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    tid = (t // tile[0], h // tile[1], w // tile[2])
                    tile_ids.append(tid)
        by_tile: dict[tuple[int, int, int], list[list[float]]] = {}
        for i in range(S):
            by_tile.setdefault(tile_ids[i], []).append(rows[i])
        for tid, group in by_tile.items():
            for row in group[1:]:
                assert row == group[0], f"tile {tid} has non-uniform rows"

    def test_window_zero_means_only_own_tile(self):
        # tile=(1,2,2), window=(0,0,0) → query attends only to its own tile.
        T, H, W = 2, 2, 2
        m = sliding_tile_block_mask(T=T, H=H, W=W, tile=(1, 2, 2), window=(0, 0, 0))[0, 0]
        grid = cast(list[list[float]], m.tolist())
        for i in range(8):
            for j in range(8):
                same_tile = (i // 4) == (j // 4)
                if same_tile:
                    assert grid[i][j] == 0.0
                else:
                    assert math.isinf(grid[i][j]) and grid[i][j] < 0

    def test_validation(self):
        with pytest.raises(ValueError):
            sliding_tile_block_mask(T=2, H=3, W=4, tile=(1, 2, 2))
        with pytest.raises(ValueError):
            sliding_tile_block_mask(T=2, H=4, W=4, tile=(1, 0, 2))
        with pytest.raises(ValueError):
            sliding_tile_block_mask(T=2, H=4, W=4, tile=(1, 2, 2), window=(0, -1, 0))


class TestRadialBoxMask:
    def test_shape(self):
        m = radial_box_mask(T=2, H=3, W=4, radius_t=1, radius_s=2.0)
        assert m.shape == (1, 1, 24, 24)

    def test_pattern(self):
        # T=2, H=3, W=3. radius_t=0 → same frame only. radius_s=1.0 → 4-neighbour (incl. self).
        T, H, W = 2, 3, 3
        m = cast(
            list[list[float]],
            radial_box_mask(T=T, H=H, W=W, radius_t=0, radius_s=1.0)[0, 0].tolist(),
        )
        for ti in range(T):
            for hi in range(H):
                for wi in range(W):
                    i = ti * H * W + hi * W + wi
                    for tj in range(T):
                        for hj in range(H):
                            for wj in range(W):
                                j = tj * H * W + hj * W + wj
                                dt = abs(ti - tj)
                                ds = math.sqrt((hi - hj) ** 2 + (wi - wj) ** 2)
                                ok = (dt <= 0) and (ds <= 1.0)
                                if ok:
                                    assert m[i][j] == 0.0
                                else:
                                    assert math.isinf(m[i][j]) and m[i][j] < 0

    def test_validation(self):
        with pytest.raises(ValueError):
            radial_box_mask(T=2, H=2, W=2, radius_t=-1, radius_s=1.0)
        with pytest.raises(ValueError):
            radial_box_mask(T=2, H=2, W=2, radius_t=1, radius_s=-0.5)
        with pytest.raises(ValueError):
            radial_box_mask(T=0, H=2, W=2, radius_t=0, radius_s=1.0)


class TestRadialGaussianMask:
    def test_shape(self):
        m = radial_gaussian_mask(T=2, H=3, W=4, sigma_t=1.0, sigma_s=1.5)
        assert m.shape == (1, 1, 24, 24)

    def test_self_is_zero(self):
        m = radial_gaussian_mask(T=2, H=2, W=2, sigma_t=1.0, sigma_s=1.0)[0, 0]
        diag = cast(list[float], mx.diagonal(m).tolist())
        for v in diag:
            assert v == 0.0

    def test_monotonic_decay(self):
        # Along temporal axis, fixed (h, w): log-weight strictly decreases with |Δt|.
        T, H, W = 4, 2, 2
        raw = radial_gaussian_mask(T=T, H=H, W=W, sigma_t=1.0, sigma_s=10.0, cutoff=-1000.0)
        m = cast(list[list[float]], raw[0, 0].tolist())
        same_pos_indices = [t * H * W for t in range(T)]
        vals = [m[0][k] for k in same_pos_indices]
        for k in range(1, len(vals)):
            assert vals[k] < vals[k - 1], f"non-monotonic at t={k}: {vals}"

    def test_cutoff_clamps_to_neg_inf(self):
        # Very tight sigmas with cutoff=-1 → everything except self gets -inf.
        m = radial_gaussian_mask(T=2, H=2, W=2, sigma_t=0.01, sigma_s=0.01, cutoff=-1.0)[0, 0]
        grid = cast(list[list[float]], m.tolist())
        for i in range(8):
            for j in range(8):
                if i == j:
                    assert grid[i][j] == 0.0
                else:
                    assert math.isinf(grid[i][j]) and grid[i][j] < 0

    def test_validation(self):
        with pytest.raises(ValueError):
            radial_gaussian_mask(T=2, H=2, W=2, sigma_t=0.0, sigma_s=1.0)
        with pytest.raises(ValueError):
            radial_gaussian_mask(T=2, H=2, W=2, sigma_t=1.0, sigma_s=-0.5)
        with pytest.raises(ValueError):
            radial_gaussian_mask(T=2, H=2, W=2, sigma_t=1.0, sigma_s=1.0, cutoff=0.0)
        with pytest.raises(ValueError):
            radial_gaussian_mask(T=0, H=2, W=2, sigma_t=1.0, sigma_s=1.0)
