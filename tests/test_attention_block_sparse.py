"""Tests for mlx_arsenal.attention.block_sparse and centroid_threshold_block_mask."""

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.attention import (
    block_contiguous_permutation,
    block_sparse_attention,
    block_sparse_kernel_available,
    centroid_compensated_attention,
    centroid_threshold_block_mask,
    invert_permutation,
    sliding_tile_block_mask,
    tile_labels,
)

NEG_INF = float("-inf")


def _qkv(
    B: int, H: int, Sq: int, Sk: int, D: int, seed: int
) -> tuple[mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    q, k, v = (
        array_from_any(rng.normal(size=(B, H, S, D)).astype(np.float32)) for S in (Sq, Sk, Sk)
    )
    return q, k, v


def _random_block_mask(shape: tuple[int, ...], seed: int, p_skip: float = 0.5) -> mx.array:
    rng = np.random.default_rng(seed)
    return array_from_any(np.where(rng.random(shape) < p_skip, NEG_INF, 0.0).astype(np.float32))


def _close(a: mx.array, b: mx.array, atol: float = 1e-5) -> bool:
    return np.allclose(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)), atol=atol)


class TestBlockSparseAttention:
    @pytest.mark.parametrize(("Sq", "Sk", "bs"), [(64, 64, 16), (70, 53, 16), (37, 37, 8)])
    @pytest.mark.parametrize("lead", [(1, 1), (2, 3), (2, 1), (1, 3)])
    def test_matches_centroid_compensated_attention(self, Sq, Sk, bs, lead):
        B, H = 2, 3
        q, k, v = _qkv(B, H, Sq, Sk, 8, seed=Sq + Sk)
        Cq, Ck = -(-Sq // bs), -(-Sk // bs)
        mask = _random_block_mask((*lead, Cq, Ck), seed=Sq)
        out = block_sparse_attention(q, k, v, block_mask=mask, block_size=bs)
        ref = centroid_compensated_attention(
            q,
            k,
            v,
            q_labels=mx.arange(Sq) // bs,
            k_labels=mx.arange(Sk) // bs,
            block_mask=mask,
        )
        assert out.shape == (B, H, Sq, 8)
        assert _close(out, ref)

    def test_partial_last_block_uses_its_true_length(self):
        # One query block attends to block 0 exactly; block 1 (3 keys) is a centroid
        # with a log(3) bias, i.e. the same weight as its 3 keys replaced by their mean.
        q, k, v = _qkv(1, 1, 4, 7, 4, seed=1)
        mask = array_from_any(np.array([[[[0.0, NEG_INF]]]], dtype=np.float32))
        out = block_sparse_attention(q, k, v, block_mask=mask, block_size=4)
        k_rep = mx.concatenate(
            [k[:, :, :4], mx.repeat(k[:, :, 4:].mean(axis=2, keepdims=True), 3, axis=2)], axis=2
        )
        v_rep = mx.concatenate(
            [v[:, :, :4], mx.repeat(v[:, :, 4:].mean(axis=2, keepdims=True), 3, axis=2)], axis=2
        )
        ref = mx.fast.scaled_dot_product_attention(q, k_rep, v_rep, scale=0.5)
        assert _close(out, ref)

    def test_all_kept_is_dense_attention(self):
        q, k, v = _qkv(1, 2, 40, 40, 8, seed=2)
        mask = mx.zeros((1, 1, 3, 3))
        for compensate in (True, False):
            out = block_sparse_attention(
                q, k, v, block_mask=mask, block_size=16, compensate=compensate
            )
            assert _close(out, mx.fast.scaled_dot_product_attention(q, k, v, scale=8**-0.5))

    @pytest.mark.parametrize("lead", [(1, 1), (2, 3), (2, 1), (1, 3)])
    def test_without_compensation_skipped_blocks_are_dropped(self, lead):
        # Partial blocks on both sides; the reference drops key columns token by token.
        B, H, Sq, Sk, bs = 2, 3, 21, 13, 8
        q, k, v = _qkv(B, H, Sq, Sk, 4, seed=3)
        mask = _random_block_mask((*lead, 3, 2), seed=4, p_skip=0.4)
        mask = mx.concatenate([mx.zeros((*lead, 3, 1)), mask[..., 1:]], axis=-1)  # no empty row
        out = block_sparse_attention(
            q, k, v, block_mask=mask, block_size=bs, compensate=False, scale=0.3
        )
        keep = np.broadcast_to(np.array(mask) == 0, (B, H, 3, 2))
        qn, kn, vn = (np.array(x, dtype=np.float64) for x in (q, k, v))
        ref = np.zeros((B, H, Sq, 4))
        for b in range(B):
            for h in range(H):
                for i in range(Sq):
                    cols = [j for j in range(Sk) if keep[b, h, i // bs, j // bs]]
                    logits = 0.3 * (kn[b, h, cols] @ qn[b, h, i])
                    p = np.exp(logits - logits.max())
                    ref[b, h, i] = (p / p.sum()) @ vn[b, h, cols]
        assert np.allclose(np.array(out), ref, atol=1e-5)

    def test_value_dim_may_differ(self):
        q, k, _ = _qkv(1, 2, 30, 30, 8, seed=18)
        v = _qkv(1, 2, 30, 30, 16, seed=19)[2]
        mask = _random_block_mask((1, 2, 4, 4), seed=20)
        out = block_sparse_attention(q, k, v, block_mask=mask, block_size=8)
        ref = centroid_compensated_attention(
            q, k, v, q_labels=mx.arange(30) // 8, k_labels=mx.arange(30) // 8, block_mask=mask
        )
        assert out.shape == (1, 2, 30, 16)
        assert _close(out, ref)

    @pytest.mark.parametrize("compensate", [True, False])
    def test_only_exact_zero_keeps_a_block(self, compensate):
        q, k, v = _qkv(1, 1, 16, 16, 4, seed=5)
        skip = mx.array([[[[0.0, NEG_INF], [NEG_INF, 0.0]]]])
        for odd in ([[0.0, 1.0], [float("nan"), 0.0]], [[0.0, 1.0], [-5.0, 0.0]]):
            a = block_sparse_attention(
                q, k, v, block_mask=skip, block_size=8, compensate=compensate
            )
            b = block_sparse_attention(
                q, k, v, block_mask=mx.array([[odd]]), block_size=8, compensate=compensate
            )
            assert _close(a, b, atol=0.0)

    @pytest.mark.parametrize("dtype", [mx.bool_, mx.uint8, mx.int32])
    def test_rejects_non_floating_masks(self, dtype):
        # A "True = keep" route would otherwise read inverted (False == 0 keeps).
        q, k, v = _qkv(1, 1, 16, 16, 4, seed=21)
        with pytest.raises(ValueError, match="floating"):
            block_sparse_attention(
                q, k, v, block_mask=mx.ones((1, 1, 2, 2), dtype=dtype), block_size=8
            )

    def test_runs_under_compile(self):
        # No host reads: mx.compile rejects any .item() during tracing.
        q, k, v = _qkv(2, 2, 32, 32, 8, seed=6)
        mask = _random_block_mask((2, 2, 4, 4), seed=7)
        for compensate in (True, False):
            fn = mx.compile(
                lambda q, k, v, m, c=compensate: block_sparse_attention(
                    q, k, v, block_mask=m, block_size=8, compensate=c
                )
            )
            # Without compensation every query block needs a kept block: keep the diagonal.
            m = mask if compensate else mx.where(mx.eye(4, dtype=mx.bool_), 0.0, mask)
            ref = block_sparse_attention(q, k, v, block_mask=m, block_size=8, compensate=compensate)
            assert _close(fn(q, k, v, m), ref)

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_keeps_input_dtype(self, dtype):
        q, k, v = (x.astype(dtype) for x in _qkv(1, 1, 32, 32, 8, seed=8))
        out = block_sparse_attention(
            q, k, v, block_mask=_random_block_mask((1, 1, 2, 2), 9), block_size=16
        )
        assert out.dtype == dtype

    def test_scale_default(self):
        q, k, v = _qkv(1, 1, 16, 16, 16, seed=10)
        mask = _random_block_mask((1, 1, 2, 2), 11)
        a = block_sparse_attention(q, k, v, block_mask=mask, block_size=8)
        b = block_sparse_attention(q, k, v, block_mask=mask, block_size=8, scale=0.25)
        assert _close(a, b, atol=0.0)

    def test_kernel_not_available_yet(self):
        assert block_sparse_kernel_available(mx.float16, 128) is False

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"q": mx.zeros((2, 16, 4))}, "rank 4"),
            ({"k": mx.zeros((1, 2, 16, 4))}, "share"),
            ({"v": mx.zeros((1, 1, 15, 4))}, "key length"),
            ({"k": mx.zeros((1, 1, 16, 5)), "v": mx.zeros((1, 1, 16, 5))}, "head dim"),
            ({"q": mx.zeros((1, 1, 0, 4))}, "non-empty"),
            ({"block_size": 0}, "block_size"),
            ({"block_mask": mx.zeros((2, 2))}, "rank 4"),
            ({"block_mask": mx.zeros((1, 1, 2, 3))}, "does not broadcast"),
            ({"block_mask": mx.zeros((2, 1, 2, 2))}, "does not broadcast"),
            ({"block_mask": mx.zeros((1, 2, 2, 2))}, "does not broadcast"),
        ],
    )
    def test_rejects_bad_shapes(self, kwargs, match):
        args: dict[str, Any] = {
            "q": mx.zeros((1, 1, 16, 4)),
            "k": mx.zeros((1, 1, 16, 4)),
            "v": mx.zeros((1, 1, 16, 4)),
            "block_mask": mx.zeros((1, 1, 2, 2)),
            "block_size": 8,
        }
        args.update(kwargs)
        q, k, v = args.pop("q"), args.pop("k"), args.pop("v")
        with pytest.raises(ValueError, match=match):
            block_sparse_attention(q, k, v, **args)


class TestPermutedBlocks:
    """ADR-0001 exit clause: permutation and masks at the kernel's block granularity."""

    def test_tiles_permuted_into_blocks_match_tile_clusters(self):
        # A (2, 16, 16) latent grid tiled (1, 8, 8): every tile is 64 tokens, so after
        # sorting tokens by tile each tile is exactly one 64-token block.
        T, Hh, Ww, tile = 2, 16, 16, (1, 8, 8)
        q, k, v = _qkv(1, 2, T * Hh * Ww, T * Hh * Ww, 8, seed=12)
        labels = tile_labels(T, Hh, Ww, tile=tile)
        tile_mask = sliding_tile_block_mask(T, Hh // 8, Ww // 8, tile=(1, 1, 1), window=(0, 0, 0))
        perm, inv = block_contiguous_permutation(-labels.astype(mx.float32), block_size=64)
        assert np.array_equal(np.array(labels[perm]), np.repeat(np.arange(8), 64))
        out = block_sparse_attention(
            q[:, :, perm], k[:, :, perm], v[:, :, perm], block_mask=tile_mask, block_size=64
        )[:, :, inv]
        ref = centroid_compensated_attention(
            q, k, v, q_labels=labels, k_labels=labels, block_mask=tile_mask
        )
        assert _close(out, ref)
        assert np.array_equal(np.array(invert_permutation(perm)), np.array(inv))

    def test_router_mask_drives_permuted_attention(self):
        q, k, v = _qkv(1, 2, 256, 256, 16, seed=13)
        perm, inv = block_contiguous_permutation(q[0, 0, :, 0], block_size=64)
        qp, kp, vp = q[:, :, perm], k[:, :, perm], v[:, :, perm]
        mask = centroid_threshold_block_mask(qp, kp, tau=0.5, band=0)
        out = block_sparse_attention(qp, kp, vp, block_mask=mask)[:, :, inv]
        labels = inv // 64  # block of every original token after the permutation
        ref = centroid_compensated_attention(
            q, k, v, q_labels=labels, k_labels=labels, block_mask=mask
        )
        assert _close(out, ref)


def _threshold_oracle(q, k, tau, bs, band, scale):
    """NumPy float64 restatement of the Sol 'diag' rule."""
    qn, kn = (np.array(x.astype(mx.float32), dtype=np.float64) for x in (q, k))

    def means(x):
        N = x.shape[2]
        return np.stack([x[:, :, i : i + bs].mean(axis=2) for i in range(0, N, bs)], axis=2)

    qc, kc = means(qn), means(kn)
    c = scale * math.log2(math.e)
    mu, var = kc.mean(axis=2, keepdims=True), kc.var(axis=2, keepdims=True)
    thr = (qc * mu).sum(-1) * c + tau * np.sqrt((qc * qc * var).sum(-1) * c * c + 1e-6)
    scores = np.einsum("bhid,bhjd->bhij", qc, kc) * c
    i, j = np.arange(qc.shape[2]), np.arange(kc.shape[2])
    keep = (scores > thr[..., None]) | (np.abs(i[:, None] - j[None, :]) <= band)
    return keep


class TestCentroidThresholdBlockMask:
    @pytest.mark.parametrize(("Nq", "Nk", "bs"), [(256, 256, 32), (250, 197, 32), (90, 77, 16)])
    @pytest.mark.parametrize("tau", [0.0, 1.0, 1.5])
    @pytest.mark.parametrize("band", [0, 1])
    def test_matches_oracle(self, Nq, Nk, bs, tau, band):
        q, k, _ = _qkv(2, 3, Nq, Nk, 16, seed=Nq + Nk)
        q, k = 3 * q, 3 * k  # spread the block scores
        out = np.array(centroid_threshold_block_mask(q, k, tau=tau, block_size=bs, band=band))
        keep = _threshold_oracle(q, k, tau, bs, band, 0.25)
        assert out.shape == keep.shape
        assert np.isin(out, (0.0, -np.inf)).all()
        assert np.array_equal(out == 0, keep)

    def test_neg_inf_keeps_everything(self):
        q, k, _ = _qkv(1, 2, 100, 70, 8, seed=14)
        out = centroid_threshold_block_mask(q, k, tau=NEG_INF, block_size=32, band=0)
        assert out.shape == (1, 2, 4, 3)
        assert np.all(np.array(out) == 0)

    def test_band_is_always_kept(self):
        q, k, _ = _qkv(1, 1, 128, 128, 8, seed=15)
        for band in (0, 2):
            out = np.array(centroid_threshold_block_mask(q, k, tau=1e9, block_size=16, band=band))
            i = np.arange(8)
            assert np.array_equal(out[0, 0] == 0, np.abs(i[:, None] - i[None, :]) <= band)

    def test_keeps_a_distant_matching_block(self):
        # Query block 0 and key block 6 share a strong direction no other block has.
        rng = np.random.default_rng(16)
        q = rng.normal(size=(1, 1, 128, 16)).astype(np.float32)
        k = rng.normal(size=(1, 1, 128, 16)).astype(np.float32)
        q[0, 0, :16, 0] += 8.0
        k[0, 0, 96:112, 0] += 8.0
        out = np.array(
            centroid_threshold_block_mask(
                array_from_any(q), array_from_any(k), tau=1.0, block_size=16
            )
        )
        assert out[0, 0, 0, 6] == 0
        assert (out[0, 0, 0, 2:6] == NEG_INF).all()

    def test_ties_with_the_threshold_are_skipped(self):
        # q = 0: every score equals the threshold at tau = 0; only the band survives.
        _, k, _ = _qkv(1, 1, 64, 64, 8, seed=22)
        out = np.array(
            centroid_threshold_block_mask(
                mx.zeros((1, 1, 64, 8)), k, tau=0.0, block_size=16, band=0
            )
        )
        assert np.array_equal(out[0, 0] == 0, np.eye(4, dtype=bool))

    def test_floor_under_the_square_root(self):
        # q = 0 gives a zero std: the 1e-6 floor puts the threshold just below 0 at tau < 0.
        _, k, _ = _qkv(1, 1, 64, 64, 8, seed=23)
        out = np.array(
            centroid_threshold_block_mask(
                mx.zeros((1, 1, 64, 8)), k, tau=-1.0, block_size=16, band=0
            )
        )
        assert np.all(out == 0)

    def test_scale_default(self):
        q, k, _ = _qkv(1, 2, 128, 128, 16, seed=17)
        a = centroid_threshold_block_mask(3 * q, 3 * k, tau=1.0, block_size=16)
        b = centroid_threshold_block_mask(3 * q, 3 * k, tau=1.0, block_size=16, scale=0.25)
        assert np.array_equal(np.array(a), np.array(b))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"q": mx.zeros((2, 16, 4))}, "rank 4"),
            ({"k": mx.zeros((1, 2, 16, 4))}, "share"),
            ({"k": mx.zeros((1, 1, 16, 5))}, "head dim"),
            ({"k": mx.zeros((1, 1, 0, 4))}, "non-empty"),
            ({"block_size": 0}, "block_size"),
            ({"band": -1}, "band"),
            ({"tau": math.inf}, "tau"),
            ({"tau": math.nan}, "tau"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, match):
        args: dict[str, Any] = {
            "q": mx.zeros((1, 1, 16, 4)),
            "k": mx.zeros((1, 1, 16, 4)),
            "tau": 1.0,
            "block_size": 8,
        }
        args.update(kwargs)
        q, k = args.pop("q"), args.pop("k")
        with pytest.raises(ValueError, match=match):
            centroid_threshold_block_mask(q, k, **args)
