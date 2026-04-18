"""Tests for layout module."""

import mlx.core as mx

from mlx_arsenal.layout import (
    channels_last,
    convert_conv_weights,
    to_channels_first,
    to_channels_last,
)


class TestChannelConversion:
    def test_3d_roundtrip(self):
        x = mx.random.normal((2, 3, 10))  # BCL
        cl = to_channels_last(x)
        mx.eval(cl)
        assert cl.shape == (2, 10, 3)
        cf = to_channels_first(cl)
        mx.eval(cf)
        assert mx.allclose(x, cf, atol=1e-6)

    def test_4d_roundtrip(self):
        x = mx.random.normal((2, 3, 8, 8))  # BCHW
        cl = to_channels_last(x)
        mx.eval(cl)
        assert cl.shape == (2, 8, 8, 3)
        cf = to_channels_first(cl)
        mx.eval(cf)
        assert mx.allclose(x, cf, atol=1e-6)

    def test_5d_roundtrip(self):
        x = mx.random.normal((1, 3, 4, 8, 8))  # BCDHW
        cl = to_channels_last(x)
        mx.eval(cl)
        assert cl.shape == (1, 4, 8, 8, 3)
        cf = to_channels_first(cl)
        mx.eval(cf)
        assert mx.allclose(x, cf, atol=1e-6)


class TestChannelsLastContext:
    def test_context_manager(self):
        x = mx.random.normal((1, 3, 8, 8))  # NCHW
        ref = [x]
        with channels_last(ref):
            assert ref[0].shape == (1, 8, 8, 3)  # NHWC
        assert ref[0].shape == (1, 3, 8, 8)  # Back to NCHW


class TestConvertConvWeights:
    def test_conv1d(self):
        w = mx.random.normal((8, 4, 3))  # (O, I, K)
        converted = convert_conv_weights(w)
        mx.eval(converted)
        assert converted.shape == (8, 3, 4)  # (O, K, I)

    def test_conv2d(self):
        w = mx.random.normal((8, 4, 3, 3))  # (O, I, kH, kW)
        converted = convert_conv_weights(w)
        mx.eval(converted)
        assert converted.shape == (8, 3, 3, 4)  # (O, kH, kW, I)

    def test_conv3d(self):
        w = mx.random.normal((8, 4, 3, 3, 3))  # (O, I, kD, kH, kW)
        converted = convert_conv_weights(w)
        mx.eval(converted)
        assert converted.shape == (8, 3, 3, 3, 4)  # (O, kD, kH, kW, I)
