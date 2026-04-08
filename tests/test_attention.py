"""Tests for attention module."""

import mlx.core as mx
import pytest

from mlx_ops.attention import (
    sdpa,
    SDPAttention,
    causal_mask,
    sliding_window_mask,
    RoPE2d,
    RoPE3d,
    apply_rope,
)


class TestSDPA:
    def test_basic_4d(self):
        B, H, L, D = 2, 4, 10, 32
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        out = sdpa(q, k, v)
        mx.eval(out)
        assert out.shape == (B, H, L, D)

    def test_3d_reshape(self):
        B, L, n_heads, D = 2, 10, 4, 32
        q = mx.random.normal((B, L, n_heads * D))
        k = mx.random.normal((B, L, n_heads * D))
        v = mx.random.normal((B, L, n_heads * D))
        out = sdpa(q, k, v, n_heads=n_heads)
        mx.eval(out)
        assert out.shape == (B, L, n_heads * D)

    def test_gqa(self):
        B, L, n_heads, n_kv_heads, D = 2, 10, 8, 2, 32
        q = mx.random.normal((B, L, n_heads * D))
        k = mx.random.normal((B, L, n_kv_heads * D))
        v = mx.random.normal((B, L, n_kv_heads * D))
        out = sdpa(q, k, v, n_heads=n_heads, n_kv_heads=n_kv_heads)
        mx.eval(out)
        assert out.shape == (B, L, n_heads * D)

    def test_with_mask(self):
        B, H, L, D = 1, 2, 5, 16
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        mask = causal_mask(L)
        out = sdpa(q, k, v, mask=mask)
        mx.eval(out)
        assert out.shape == (B, H, L, D)


class TestSDPAttention:
    def test_self_attention(self):
        attn = SDPAttention(dim=64, n_heads=4)
        x = mx.random.normal((2, 10, 64))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_cross_attention(self):
        attn = SDPAttention(dim=64, n_heads=4)
        x = mx.random.normal((2, 10, 64))
        kv = mx.random.normal((2, 20, 64))
        out = attn(x, kv=kv)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_gqa_attention(self):
        attn = SDPAttention(dim=64, n_heads=8, n_kv_heads=2)
        x = mx.random.normal((2, 10, 64))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)


class TestMasks:
    def test_causal_mask_shape(self):
        mask = causal_mask(5)
        mx.eval(mask)
        assert mask.shape == (1, 1, 5, 5)

    def test_causal_mask_values(self):
        mask = causal_mask(3)
        mx.eval(mask)
        # Position 0 can only see position 0
        assert mask[0, 0, 0, 1].item() == float("-inf")
        assert mask[0, 0, 0, 0].item() == 0.0
        # Position 2 can see all positions
        assert mask[0, 0, 2, 0].item() == 0.0
        assert mask[0, 0, 2, 2].item() == 0.0

    def test_causal_mask_with_offset(self):
        mask = causal_mask(3, offset=2)
        mx.eval(mask)
        assert mask.shape == (1, 1, 3, 5)
        # Position 0 (with offset 2) can see positions 0, 1, 2
        assert mask[0, 0, 0, 2].item() == 0.0

    def test_sliding_window_mask(self):
        mask = sliding_window_mask(5, window_size=2)
        mx.eval(mask)
        assert mask.shape == (1, 1, 5, 5)
        # Position 3 can see positions 2 and 3 (window=2)
        assert mask[0, 0, 3, 3].item() == 0.0
        assert mask[0, 0, 3, 2].item() == 0.0
        assert mask[0, 0, 3, 1].item() == float("-inf")


class TestRoPE2d:
    def test_output_shape_4d(self):
        rope = RoPE2d(dim=64, max_h=16, max_w=16)
        x = mx.random.normal((2, 16, 4, 64))  # (B, N=4*4, H, D)
        out = rope(x, h=4, w=4)
        mx.eval(out)
        assert out.shape == x.shape

    def test_output_shape_3d(self):
        rope = RoPE2d(dim=64, max_h=16, max_w=16)
        x = mx.random.normal((2, 16, 64))  # (B, N=4*4, D)
        out = rope(x, h=4, w=4)
        mx.eval(out)
        assert out.shape == x.shape


class TestRoPE3d:
    def test_output_shape_4d(self):
        rope = RoPE3d(dim=64, max_t=8, max_h=8, max_w=8)
        x = mx.random.normal((2, 8, 4, 64))  # (B, N=2*2*2, H, D)
        out = rope(x, t=2, h=2, w=2)
        mx.eval(out)
        assert out.shape == x.shape

    def test_output_shape_3d(self):
        rope = RoPE3d(dim=64, max_t=8, max_h=8, max_w=8)
        x = mx.random.normal((2, 8, 64))
        out = rope(x, t=2, h=2, w=2)
        mx.eval(out)
        assert out.shape == x.shape

    def test_different_positions_differ(self):
        rope = RoPE3d(dim=64, max_t=8, max_h=8, max_w=8)
        x = mx.ones((1, 8, 64))
        out = rope(x, t=2, h=2, w=2)
        mx.eval(out)
        # Different positions should get different embeddings
        assert not mx.allclose(out[0, 0], out[0, 1], atol=1e-5)


class TestApplyRope:
    def test_basic(self):
        x = mx.random.normal((2, 10, 64))
        cos = mx.ones((2, 10, 32))
        sin = mx.zeros((2, 10, 32))
        out = apply_rope(x, cos, sin)
        mx.eval(out)
        # cos=1, sin=0 should be identity
        assert mx.allclose(out, x, atol=1e-5)
