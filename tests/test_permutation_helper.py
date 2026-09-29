"""Tests for the private scatter-based permutation inverse."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._permutation import invert_permutation_last_axis


@pytest.mark.parametrize("shape", [(7,), (3, 16), (2, 3, 4, 33)])
def test_matches_argsort(shape):
    rng = np.random.default_rng(0)
    perm = np.argsort(rng.standard_normal(shape), axis=-1)
    got = invert_permutation_last_axis(mx.array(perm))
    np.testing.assert_array_equal(np.array(got), np.argsort(perm, axis=-1))
    assert got.dtype == mx.int32


def test_scalar_raises():
    with pytest.raises(ValueError):
        invert_permutation_last_axis(mx.array(3))


def test_public_invert_permutation_stays_a_permutation_on_bad_input():
    # The public 1-D helper keeps argsort semantics: garbage in still yields a
    # permutation (the scatter path is reserved for internal, valid orders).
    from mlx_arsenal.attention import invert_permutation

    for bad in ([0, 0, 2], [0, 5, 1]):
        out = sorted(np.array(invert_permutation(mx.array(bad))).tolist())
        assert out == [0, 1, 2]
