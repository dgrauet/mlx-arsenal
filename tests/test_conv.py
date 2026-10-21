"""Tests for conv module (weight normalization)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_arsenal.conv import CausalConv3d, weight_norm


class TestWeightNorm:
    def test_linear(self):
        linear = nn.Linear(8, 16)
        wn = weight_norm(linear)
        x = mx.random.normal((2, 8))
        out = wn(x)
        mx.eval(out)
        assert out.shape == (2, 16)

    def test_conv1d(self):
        conv = nn.Conv1d(4, 8, 3, padding=1)
        wn = weight_norm(conv)
        x = mx.random.normal((2, 10, 4))
        out = wn(x)
        mx.eval(out)
        assert out.shape == (2, 10, 8)

    def test_weight_normalized(self):
        """After applying weight norm, direction should be unit norm."""
        linear = nn.Linear(8, 4)
        wn = weight_norm(linear, dim=0)
        w = wn._compute_weight()
        mx.eval(w)
        norms = mx.sqrt(mx.sum(w * w, axis=1))
        expected = wn.g.squeeze()
        mx.eval(norms, expected)
        assert mx.allclose(norms, expected, atol=1e-5).item()


class TestWeightNormStateIntegrity:
    def test_wrapped_weight_not_shadowed_after_call(self):
        lin = nn.Linear(4, 4)
        wn = weight_norm(lin)
        wn(mx.ones((1, 4)))
        # The wrapper must not leave an instance-dict entry that shadows the
        # module's parameter dict (it would freeze `lin.weight` forever).
        assert "weight" not in lin.__dict__

    def test_update_after_call_is_visible(self):
        lin = nn.Linear(4, 4)
        wn = weight_norm(lin)
        wn(mx.ones((1, 4)))
        new_w = mx.zeros((4, 4))
        lin.update({"weight": new_w})
        assert mx.array_equal(lin.weight, new_w).item()

    def test_output_matches_reference(self):
        lin = nn.Linear(4, 3)
        wn = weight_norm(lin)
        x = mx.random.normal((2, 4))
        v = wn.v
        g = wn.g
        norm = mx.sqrt(mx.sum(v * v, axis=1, keepdims=True) + 1e-12)
        expected = x @ (g * (v / norm)).T + lin.bias
        assert mx.allclose(wn(x), expected, atol=1e-6).item()

    def test_weight_restored_after_module_raises(self):
        class Boom(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mx.ones((2, 2))

            def __call__(self, x):
                raise RuntimeError("boom")

        boom = Boom()
        original = boom.weight
        wn = weight_norm(boom)
        with pytest.raises(RuntimeError, match="boom"):
            wn(mx.ones((1, 2)))
        assert mx.array_equal(boom.weight, original).item()
        assert "weight" not in boom.__dict__


class TestWeightNormVariants:
    def test_dim_1_normalizes_over_other_axes(self):
        """dim=1 keeps per-input-channel magnitudes: norm over axis 0."""
        lin = nn.Linear(4, 3)
        wn = weight_norm(lin, dim=1)
        assert wn.g.shape == (1, 4)
        x = mx.random.normal((2, 4))
        v = wn.v
        norm = mx.sqrt(mx.sum(v * v, axis=0, keepdims=True) + 1e-12)
        expected = x @ (wn.g * (v / norm)).T + lin.bias
        assert mx.allclose(wn(x), expected, atol=1e-6).item()

    def test_custom_weight_name(self):
        class Kernelized(nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel = mx.random.normal((3, 4))

            def __call__(self, x: mx.array) -> mx.array:
                return x @ self.kernel.T

        mod = Kernelized()
        original = mod.kernel
        wn = weight_norm(mod, weight_name="kernel")
        x = mx.random.normal((2, 4))
        norm = mx.sqrt(mx.sum(wn.v * wn.v, axis=1, keepdims=True) + 1e-12)
        expected = x @ (wn.g * (wn.v / norm)).T
        assert mx.allclose(wn(x), expected, atol=1e-6).item()
        # Wrapped param untouched after the call.
        assert mx.array_equal(mod.kernel, original).item()

    def test_reentrant_calls_identical_and_side_effect_free(self):
        lin = nn.Linear(4, 3)
        wn = weight_norm(lin)
        original = lin.weight
        x = mx.random.normal((2, 4))
        out1 = wn(x)
        out2 = wn(x)
        assert mx.allclose(out1, out2, atol=1e-6).item()
        assert mx.array_equal(lin.weight, original).item()
        assert "weight" not in lin.__dict__


class TestCausalConv3d:
    @staticmethod
    def _make(kernel=3, dilation=1, pad_mode="zeros", **kw):
        mx.random.seed(0)
        conv = CausalConv3d(4, 6, kernel, dilation=dilation, pad_mode=pad_mode, **kw)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.1
        conv.bias = mx.random.normal(conv.bias.shape) * 0.1
        return conv

    @staticmethod
    def _video(T=7, H=5, W=4, C=4, seed=1):
        return mx.random.normal((2, T, H, W, C), key=mx.random.key(seed))

    def test_matches_left_padded_conv3d(self):
        conv = self._make(kernel=3, padding=1)
        x = self._video()
        padded = mx.pad(x, [(0, 0), (2, 0), (0, 0), (0, 0), (0, 0)])
        ref = mx.conv3d(padded, conv.weight, padding=(0, 1, 1)) + conv.bias
        out = conv(x)
        assert out.shape == (2, 7, 5, 4, 6)
        assert mx.allclose(out, ref, atol=1e-5)

    def test_replicate_pads_with_first_frame(self):
        conv = self._make(kernel=3, padding=1, pad_mode="replicate")
        x = self._video()
        padded = mx.concatenate([x[:, :1], x[:, :1], x], axis=1)
        ref = mx.conv3d(padded, conv.weight, padding=(0, 1, 1)) + conv.bias
        assert mx.allclose(conv(x), ref, atol=1e-5)

    @pytest.mark.parametrize("pad_mode", ["zeros", "replicate"])
    def test_causal(self, pad_mode):
        conv = self._make(kernel=3, dilation=2, padding=1, pad_mode=pad_mode)
        x = self._video(T=9)
        y = conv(x)
        x2 = x.at[:, 5:].add(10.0)
        y2 = conv(x2)
        assert mx.allclose(y[:, :5], y2[:, :5], atol=1e-5)
        assert not mx.allclose(y[:, 5:], y2[:, 5:], atol=1e-3)

    @pytest.mark.parametrize("pad_mode", ["zeros", "replicate"])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("chunks", [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 4, 4], [2, 3, 4], [9]])
    def test_step_chunks_match_one_pass(self, pad_mode, dilation, chunks):
        conv = self._make(kernel=3, dilation=dilation, padding=1, pad_mode=pad_mode)
        x = self._video(T=sum(chunks))
        full = conv(x)
        cache = None
        outs = []
        start = 0
        for n in chunks:
            y, cache = conv.step(x[:, start : start + n], cache)
            outs.append(y)
            start += n
        assert cache is not None and cache.shape[1] == conv.context_frames
        assert mx.allclose(mx.concatenate(outs, axis=1), full, atol=1e-5)

    def test_kernel_one_has_no_context(self):
        conv = self._make(kernel=1)
        assert conv.context_frames == 0
        x = self._video()
        y, cache = conv.step(x[:, :3])
        assert cache is None
        assert mx.allclose(y, conv(x[:, :3]), atol=1e-6)

    def test_spatial_stride_and_no_bias(self):
        conv = CausalConv3d(4, 6, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1), bias=False)
        assert "bias" not in conv
        assert conv(self._video(H=8, W=8)).shape == (2, 7, 4, 4, 6)

    def test_temporal_stride_one_pass(self):
        conv = CausalConv3d(4, 6, 3, stride=(2, 1, 1), padding=1)
        assert conv(self._video(T=9)).shape == (2, 5, 5, 4, 6)

    def test_step_rejects_temporal_stride(self):
        conv = CausalConv3d(4, 6, 3, stride=(2, 1, 1), padding=1)
        with pytest.raises(ValueError, match="temporal stride"):
            conv.step(self._video())

    def test_rejects_oversized_cache(self):
        conv = self._make(kernel=3, padding=1)
        with pytest.raises(ValueError, match="cache"):
            conv(self._video(), cache=self._video(T=3))

    def test_rejects_unknown_pad_mode(self):
        with pytest.raises(ValueError, match="pad_mode"):
            CausalConv3d(4, 6, 3, pad_mode="reflect")  # ty: ignore[invalid-argument-type]

    def test_weight_layout_and_keys(self):
        conv = CausalConv3d(4, 6, (3, 1, 1))
        assert conv.weight.shape == (6, 3, 1, 1, 4)
        assert set(dict(conv.parameters())) == {"weight", "bias"}
