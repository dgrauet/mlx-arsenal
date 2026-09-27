"""Tests for mlx_arsenal.attention.compensation."""

import mlx.core as mx
import pytest

from mlx_arsenal.attention import sliding_tile_block_mask, tile_labels


class TestTileLabels:
    def test_values_on_small_grid(self):
        # T=2, H=2, W=4 with tile (1, 2, 2): grid (2, 1, 2) -> 4 tiles.
        labels = tile_labels(2, 2, 4, tile=(1, 2, 2))
        assert labels.dtype == mx.int32
        assert labels.tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]

    @pytest.mark.parametrize(
        ("thw", "tile", "window"),
        [((2, 4, 4), (1, 2, 2), (1, 1, 1)), ((4, 4, 6), (2, 2, 3), (0, 1, 0))],
    )
    def test_expands_tile_level_sta_to_token_level(self, thw, tile, window):
        T, H, W = thw
        tt, th, tw = tile
        labels = tile_labels(T, H, W, tile=tile)
        grid = sliding_tile_block_mask(T // tt, H // th, W // tw, tile=(1, 1, 1), window=window)
        expanded = mx.take(mx.take(grid[0, 0], labels, axis=0), labels, axis=1)
        ref = sliding_tile_block_mask(T, H, W, tile=tile, window=window)[0, 0]
        assert mx.array_equal(expanded, ref).item()

    def test_validation(self):
        with pytest.raises(ValueError):
            tile_labels(0, 2, 2, tile=(1, 1, 1))
        with pytest.raises(ValueError):
            tile_labels(2, 2, 2, tile=(0, 1, 1))
        with pytest.raises(ValueError):
            tile_labels(2, 3, 2, tile=(1, 2, 1))
