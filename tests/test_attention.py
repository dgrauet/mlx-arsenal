"""Tests for attention module (masks)."""

import math
from typing import cast

import mlx.core as mx
import pytest

from mlx_arsenal.attention import causal_mask, sliding_window_mask, spatial_only_mask


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
