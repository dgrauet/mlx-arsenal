"""Tests for skip_error_block_scores (Triage's router score)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal.attention import (
    block_sparse_attention,
    skip_error_block_scores,
    top_k_block_mask,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def triage_reference(q, k, v, block_size, energy=None, scale=None):
    """Float64 transcription of Triage's appendix (one cluster per key block)."""
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    s = 1.0 / math.sqrt(D) if scale is None else scale
    qs = [slice(i, min(i + block_size, Nq)) for i in range(0, Nq, block_size)]
    ks = [slice(j, min(j + block_size, Nk)) for j in range(0, Nk, block_size)]
    out = np.zeros((B, H, len(qs), len(ks)))
    for b in range(B):
        if energy is None:
            e = np.array([np.sum(v[b, h].mean(0) ** 2) for h in range(H)])
        else:
            e = np.broadcast_to(energy, (B, H))[b]
        w = e / e.mean()
        for h in range(H):
            K, V = k[b, h], v[b, h]
            n = np.array([K[c].shape[0] for c in ks], dtype=np.float64)
            kbar = np.stack([K[c].mean(0) for c in ks])
            vbar = np.stack([V[c].mean(0) for c in ks])
            tr = np.array([np.mean(np.sum((K[c] - K[c].mean(0)) ** 2, -1)) for c in ks])
            fro = np.array(
                [
                    np.linalg.norm((K[c] - K[c].mean(0)).T @ (V[c] - V[c].mean(0))) / len(K[c])
                    for c in ks
                ]
            )
            for i, ci in enumerate(qs):
                Q = q[b, h, ci]
                qbar = Q.mean(0)
                qsq = np.mean(np.sum(Q**2, -1))
                qvar = np.mean(np.sum((Q - qbar) ** 2, -1))
                J = 1.0 - np.exp(-(s**2) * qsq * tr / (2 * D))
                R = (
                    s
                    * math.sqrt(qsq)
                    * fro
                    / (math.sqrt(D) * np.maximum(np.linalg.norm(vbar, axis=-1), 1e-6))
                )
                logits = s * K @ qbar + (s**2) * qvar * np.sum(K**2, -1) / (2 * D)
                L = np.array([np.log(np.sum(np.exp(logits[c]))) for c in ks])
                a = _softmax(L)
                Bt = np.mean([_softmax(s * kbar @ qi + np.log(n)) for qi in Q], axis=0)
                bt = Bt / (1 - J)
                bt = bt / bt.sum()
                out[b, h, i] = 0.5 * (a + bt) * (J + R) * math.sqrt(w[h])
    return out


def _qkv(seed, B=1, H=2, Nq=96, Nk=80, D=16, Dv=8):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, Nq, D)) * 2.0
    k = rng.standard_normal((B, H, Nk, D)) * 2.0
    v = rng.standard_normal((B, H, Nk, Dv)) + 0.5
    return q, k, v


def _run(q, k, v, **kw):
    out = skip_error_block_scores(
        mx.array(q, dtype=mx.float32),
        mx.array(k, dtype=mx.float32),
        mx.array(v, dtype=mx.float32),
        **kw,
    )
    return np.array(out)


class TestReferenceParity:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("shape", [(1, 2, 96, 80, 16, 8), (2, 3, 64, 64, 8, 8)])
    def test_matches_reference(self, seed, shape):
        q, k, v = _qkv(seed, *shape)
        bs = 16
        np.testing.assert_allclose(
            _run(q, k, v, block_size=bs), triage_reference(q, k, v, bs), rtol=2e-4, atol=1e-7
        )

    def test_partial_last_blocks(self):
        q, k, v = _qkv(3, 1, 2, 70, 53, 8, 4)
        np.testing.assert_allclose(
            _run(q, k, v, block_size=16), triage_reference(q, k, v, 16), rtol=2e-4, atol=1e-7
        )

    def test_custom_scale(self):
        q, k, v = _qkv(4)
        np.testing.assert_allclose(
            _run(q, k, v, block_size=16, scale=0.1),
            triage_reference(q, k, v, 16, scale=0.1),
            rtol=2e-4,
            atol=1e-7,
        )

    @pytest.mark.parametrize("energy_shape", [(2,), (2, 2)])
    def test_head_energy(self, energy_shape):
        q, k, v = _qkv(5, B=2)
        energy = np.random.default_rng(6).uniform(0.5, 4.0, energy_shape)
        got = _run(q, k, v, block_size=16, head_energy=mx.array(energy, dtype=mx.float32))
        np.testing.assert_allclose(
            got, triage_reference(q, k, v, 16, energy=energy), rtol=2e-4, atol=1e-7
        )


class TestProperties:
    def test_constant_block_scores_zero_and_skips_exactly(self):
        # Identical keys and values in a block: its centroid represents it exactly.
        q, k, v = _qkv(7, 1, 1, 32, 48, 8, 4)
        k[0, 0, 16:32] = k[0, 0, 16]
        v[0, 0, 16:32] = v[0, 0, 16]
        err = _run(q, k, v, block_size=16)
        assert np.all(err[..., 1] < 1e-9 * err[..., [0, 2]].min())
        mask = np.zeros((1, 1, 2, 3), dtype=np.float32)
        mask[..., 1] = -np.inf
        args = [mx.array(x, dtype=mx.float32) for x in (q, k, v)]
        dense = block_sparse_attention(*args, block_mask=mx.zeros((1, 1, 2, 3)), block_size=16)
        skipped = block_sparse_attention(*args, block_mask=mx.array(mask), block_size=16)
        np.testing.assert_allclose(np.array(skipped), np.array(dense), atol=1e-5)

    def test_spread_increases_error(self):
        q, k, v = _qkv(8, 1, 1, 16, 32, 8, 4)
        base = _run(q, k, v, block_size=16)
        k2 = k.copy()
        c = k2[0, 0, 16:].mean(0)
        k2[0, 0, 16:] = c + 3.0 * (k2[0, 0, 16:] - c)
        wider = _run(q, k2, v, block_size=16)
        assert wider[0, 0, 0, 1] > base[0, 0, 0, 1]

    def test_ranks_spread_block_above_heavy_constant_block(self):
        # A block of identical keys can carry most of the mass yet cost nothing to skip.
        rng = np.random.default_rng(9)
        D = 8
        q = rng.standard_normal((1, 1, 16, D)) * 0.1 + 1.0
        k = rng.standard_normal((1, 1, 48, D))
        k[0, 0, :16] = 1.0  # aligned with q: high mass
        v = rng.standard_normal((1, 1, 48, 4))
        v[0, 0, :16] = 2.0
        err = _run(q, k, v, block_size=16)
        assert err[0, 0, 0, 0] == 0.0
        assert np.array(top_k_block_mask(mx.array(err), 1))[0, 0, 0, 0] == -np.inf

    def test_head_weight_scales_by_sqrt_energy(self):
        q, k, v = _qkv(10, 1, 2, 32, 32, 8, 4)
        e1 = _run(q, k, v, block_size=16, head_energy=mx.array([1.0, 1.0]))
        e2 = _run(q, k, v, block_size=16, head_energy=mx.array([1.0, 9.0]))
        # w = (0.2, 1.8) against (1, 1).
        np.testing.assert_allclose(e2[:, 0], e1[:, 0] * math.sqrt(0.2), rtol=1e-5)
        np.testing.assert_allclose(e2[:, 1], e1[:, 1] * math.sqrt(1.8), rtol=1e-5)

    def test_large_spread_stays_finite(self):
        q, k, v = _qkv(11, 1, 1, 32, 32, 8, 4)
        err = _run(q * 30.0, k * 30.0, v, block_size=16)
        assert np.all(np.isfinite(err))
        assert np.all(err >= 0.0)

    def test_bf16_inputs(self):
        q, k, v = _qkv(12)
        got = skip_error_block_scores(
            mx.array(q).astype(mx.bfloat16),
            mx.array(k).astype(mx.bfloat16),
            mx.array(v).astype(mx.bfloat16),
            block_size=16,
        )
        assert got.dtype == mx.float32
        np.testing.assert_allclose(np.array(got), _run(q, k, v, block_size=16), rtol=0.1, atol=1e-4)

    def test_output_shape(self):
        q, k, v = _qkv(13, 2, 3, 70, 53, 8, 4)
        assert _run(q, k, v, block_size=16).shape == (2, 3, 5, 4)


class TestValidation:
    def _args(self):
        q, k, v = _qkv(0, 1, 2, 32, 32, 8, 4)
        return mx.array(q), mx.array(k), mx.array(v)

    def test_rank(self):
        q, k, v = self._args()
        with pytest.raises(ValueError, match="rank 4"):
            skip_error_block_scores(q[0], k, v)

    def test_batch_heads(self):
        q, k, v = self._args()
        with pytest.raises(ValueError, match="share"):
            skip_error_block_scores(q, k[:, :1], v)

    def test_key_length(self):
        q, k, v = self._args()
        with pytest.raises(ValueError, match="key length"):
            skip_error_block_scores(q, k, v[:, :, :16])

    def test_head_dim(self):
        q, k, v = self._args()
        with pytest.raises(ValueError, match="head dim"):
            skip_error_block_scores(q, k[..., :4], v)

    def test_empty(self):
        q, k, v = self._args()
        with pytest.raises(ValueError, match="non-empty"):
            skip_error_block_scores(q[:, :, :0], k, v)

    def test_block_size(self):
        q, k, v = self._args()
        with pytest.raises(ValueError, match="block_size"):
            skip_error_block_scores(q, k, v, block_size=0)

    def test_head_energy_shape(self):
        q, k, v = self._args()
        with pytest.raises(ValueError, match="head_energy"):
            skip_error_block_scores(q, k, v, head_energy=mx.ones((3,)))
