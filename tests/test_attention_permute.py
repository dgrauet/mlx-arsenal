"""Tests for mlx_arsenal.attention.permute."""

import mlx.core as mx
import pytest

from mlx_arsenal.attention import block_contiguous_permutation, invert_permutation


class TestBlockContiguousPermutation:
    def test_descending_top_scores_first(self):
        scores = mx.array([3.0, 1.0, 4.0, 1.0, 5.0])
        perm, inv = block_contiguous_permutation(scores, block_size=2)
        # Highest first: 5.0 (idx 4), then 4.0 (idx 2), then 3.0 (idx 0),
        # then ties 1.0 (idx 1) and 1.0 (idx 3) in original order (stable).
        assert perm.tolist() == [4, 2, 0, 1, 3]

    def test_ascending_bottom_scores_first(self):
        scores = mx.array([3.0, 1.0, 4.0, 1.0, 5.0])
        perm, inv = block_contiguous_permutation(scores, block_size=2, descending=False)
        # Lowest first: 1.0 (idx 1) ties 1.0 (idx 3), then 3.0 (idx 0),
        # then 4.0 (idx 2), then 5.0 (idx 4).
        assert perm.tolist() == [1, 3, 0, 2, 4]

    def test_inverse_roundtrip(self):
        # x[perm][inv_perm] must equal x, for any tensor x indexed along axis 0.
        scores = mx.array([0.7, 0.2, 0.9, 0.4, 0.1, 0.8])
        perm, inv = block_contiguous_permutation(scores, block_size=2)
        x = mx.arange(6 * 4).reshape(6, 4)  # token axis = 0
        x_perm = mx.take(x, perm, axis=0)
        x_back = mx.take(x_perm, inv, axis=0)
        assert mx.array_equal(x_back, x).item()

    def test_perm_shape_and_dtype(self):
        scores = mx.zeros((128,))
        perm, inv = block_contiguous_permutation(scores, block_size=16)
        assert perm.shape == (128,)
        assert inv.shape == (128,)
        assert perm.dtype == mx.int32
        assert inv.dtype == mx.int32

    def test_validation(self):
        with pytest.raises(ValueError):
            block_contiguous_permutation(mx.zeros((4, 4)), block_size=2)  # 2D
        with pytest.raises(ValueError):
            block_contiguous_permutation(mx.zeros((4,)), block_size=0)
        with pytest.raises(ValueError):
            block_contiguous_permutation(mx.zeros((4,)), block_size=-1)
        with pytest.raises(ValueError):
            block_contiguous_permutation(mx.array([]), block_size=2)  # empty


class TestInvertPermutation:
    def test_identity(self):
        perm = mx.array([0, 1, 2, 3], dtype=mx.int32)
        inv = invert_permutation(perm)
        assert inv.tolist() == [0, 1, 2, 3]

    def test_reverse(self):
        # Reverse permutation is its own inverse (involution).
        perm = mx.array([3, 2, 1, 0], dtype=mx.int32)
        inv = invert_permutation(perm)
        assert inv.tolist() == [3, 2, 1, 0]

    def test_arbitrary_permutation(self):
        perm = mx.array([2, 0, 3, 1], dtype=mx.int32)
        inv = invert_permutation(perm)
        # perm[inv[i]] == i for all i (definition of inverse).
        assert inv.shape == (4,)
        for i in range(4):
            assert perm[inv[i].item()].item() == i
