import math

import mlx.core as mx

from mlx_arsenal.encoding import FourierEmbedder


class TestFourierEmbedder:
    def test_output_dim_with_pi(self):
        """Output dim = input_dim * (2 * num_freqs + 1) when include_pi=True."""
        emb = FourierEmbedder(num_freqs=8, input_dim=3, include_pi=True)
        x = mx.random.normal((10, 3))
        y = emb(x)
        expected_dim = 3 * (2 * 8 + 1)  # 51
        assert y.shape == (10, expected_dim), f"Expected (10, {expected_dim}), got {y.shape}"

    def test_output_dim_without_pi(self):
        """Output dim same formula, frequencies differ but shape is same."""
        emb = FourierEmbedder(num_freqs=8, input_dim=3, include_pi=False)
        x = mx.random.normal((10, 3))
        y = emb(x)
        expected_dim = 3 * (2 * 8 + 1)  # 51
        assert y.shape == (10, expected_dim)

    def test_batched_input(self):
        """Works with arbitrary leading dimensions."""
        emb = FourierEmbedder(num_freqs=4, input_dim=3)
        x = mx.random.normal((2, 100, 3))
        y = emb(x)
        expected_dim = 3 * (2 * 4 + 1)  # 27
        assert y.shape == (2, 100, expected_dim)

    def test_frequencies_are_powers_of_two(self):
        """Frequencies are 2^0, 2^1, ..., 2^(N-1)."""
        emb = FourierEmbedder(num_freqs=4, input_dim=3, include_pi=False)
        expected = mx.array([1.0, 2.0, 4.0, 8.0])
        assert mx.allclose(emb._frequencies, expected).item()

    def test_output_contains_input(self):
        """First input_dim values of output are the raw input (include_input=True)."""
        emb = FourierEmbedder(num_freqs=2, input_dim=3)
        x = mx.array([[1.0, 2.0, 3.0]])
        y = emb(x)
        mx.eval(y)
        assert mx.allclose(y[:, :3], x).item()


def test_frequencies_are_not_trainable_parameters():
    from mlx.utils import tree_flatten

    emb = FourierEmbedder()
    params = tree_flatten(emb.parameters())
    # A fixed positional encoding has no trainable state: an optimizer must
    # not touch the frequency bands, and strict checkpoint loading must not
    # expect a `frequencies` key.
    assert params == []


def test_output_matches_manual_computation():
    emb = FourierEmbedder(num_freqs=2, input_dim=1, include_pi=False, include_input=True)
    x = mx.array([[0.5]])
    out = emb(x)
    expected = mx.array([[0.5, math.sin(0.5), math.sin(1.0), math.cos(0.5), math.cos(1.0)]])
    assert mx.allclose(out, expected, atol=1e-6).item()
