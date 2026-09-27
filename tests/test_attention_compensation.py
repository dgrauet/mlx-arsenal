"""Tests for mlx_arsenal.attention.compensation."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any, item_float
from mlx_arsenal.attention import (
    centroid_compensated_attention,
    probe_residual_correction,
    select_probe_rows,
    sliding_tile_block_mask,
    tile_labels,
)

NEG_INF = float("-inf")


def _balanced_labels(S: int, C: int, seed: int) -> mx.array:
    """Labels in [0, C) with every cluster non-empty, shuffled."""
    rng = np.random.default_rng(seed)
    return array_from_any(rng.permutation(np.arange(S) % C).astype(np.int32))


def _random_block_mask(shape: tuple[int, ...], seed: int, p_skip: float = 0.5) -> mx.array:
    rng = np.random.default_rng(seed)
    skipped = rng.random(shape) < p_skip
    return array_from_any(np.where(skipped, NEG_INF, 0.0).astype(np.float32))


def _replacement_oracle(q, k, v, q_labels, k_labels, block_mask, scale):
    """Dense attention where each skipped key/value is swapped for its cluster centroid."""
    qn, kn, vn = (np.array(x.astype(mx.float32), dtype=np.float64) for x in (q, k, v))
    B, H, Sq, _ = qn.shape
    Sk = kn.shape[2]
    ql = np.broadcast_to(np.array(q_labels), (B, H, Sq))
    kl = np.broadcast_to(np.array(k_labels), (B, H, Sk))
    Cq, Ck = block_mask.shape[-2:]
    keep = np.broadcast_to(np.array(block_mask) == 0.0, (B, H, Cq, Ck))
    out = np.zeros((B, H, Sq, vn.shape[-1]))
    for b in range(B):
        for h in range(H):
            k_bar = np.zeros((Ck, kn.shape[-1]))
            v_bar = np.zeros((Ck, vn.shape[-1]))
            for c in range(Ck):
                members = kl[b, h] == c
                if members.any():
                    k_bar[c] = kn[b, h][members].mean(axis=0)
                    v_bar[c] = vn[b, h][members].mean(axis=0)
            for i in range(Sq):
                kept = keep[b, h, ql[b, h, i], kl[b, h]]
                k_rep = np.where(kept[:, None], kn[b, h], k_bar[kl[b, h]])
                v_rep = np.where(kept[:, None], vn[b, h], v_bar[kl[b, h]])
                logits = scale * (k_rep @ qn[b, h, i])
                p = np.exp(logits - logits.max())
                out[b, h, i] = (p / p.sum()) @ v_rep
    return out


def _dense(q, k, v, scale):
    return mx.softmax(scale * (q @ k.swapaxes(-1, -2)), axis=-1) @ v


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


class TestCentroidCompensatedAttention:
    B, H, Sq, Sk, D = 1, 2, 12, 10, 8
    Cq, Ck = 3, 4

    def _qkv(self, seed: int = 0):
        mx.random.seed(seed)
        q = mx.random.normal((self.B, self.H, self.Sq, self.D))
        k = mx.random.normal((self.B, self.H, self.Sk, self.D))
        v = mx.random.normal((self.B, self.H, self.Sk, self.D))
        return q, k, v

    def test_all_kept_equals_dense(self):
        q, k, v = self._qkv()
        out = centroid_compensated_attention(
            q,
            k,
            v,
            q_labels=_balanced_labels(self.Sq, self.Cq, 1),
            k_labels=_balanced_labels(self.Sk, self.Ck, 2),
            block_mask=mx.zeros((self.Cq, self.Ck)),
        )
        ref = _dense(q, k, v, 1.0 / math.sqrt(self.D))
        assert mx.allclose(out, ref, atol=1e-5).item()

    def test_matches_replacement_oracle(self):
        q, k, v = self._qkv(3)
        ql = _balanced_labels(self.Sq, self.Cq, 4)
        kl = _balanced_labels(self.Sk, self.Ck, 5)
        bm = _random_block_mask((self.Cq, self.Ck), 6)
        out = centroid_compensated_attention(
            q, k, v, q_labels=ql, k_labels=kl, block_mask=bm, scale=0.3
        )
        ref = _replacement_oracle(q, k, v, ql, kl, bm, 0.3)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_per_head_block_mask_matches_oracle(self):
        q, k, v = self._qkv(7)
        ql = _balanced_labels(self.Sq, self.Cq, 8)
        kl = _balanced_labels(self.Sk, self.Ck, 9)
        bm = _random_block_mask((1, self.H, self.Cq, self.Ck), 10)
        out = centroid_compensated_attention(q, k, v, q_labels=ql, k_labels=kl, block_mask=bm)
        ref = _replacement_oracle(q, k, v, ql, kl, bm, 1.0 / math.sqrt(self.D))
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_all_skipped_equals_centroid_attention(self):
        q, k, v = self._qkv(11)
        ql = _balanced_labels(self.Sq, self.Cq, 12)
        kl = _balanced_labels(self.Sk, self.Ck, 13)
        bm = mx.full((self.Cq, self.Ck), NEG_INF)
        out = centroid_compensated_attention(q, k, v, q_labels=ql, k_labels=kl, block_mask=bm)
        onehot = np.eye(self.Ck)[np.array(kl)]  # (Sk, Ck)
        n = onehot.sum(axis=0)
        k_bar = np.einsum("sc,bhsd->bhcd", onehot, np.array(k)) / n[:, None]
        v_bar = np.einsum("sc,bhsd->bhcd", onehot, np.array(v)) / n[:, None]
        logits = np.einsum("bhqd,bhcd->bhqc", np.array(q), k_bar) / math.sqrt(self.D) + np.log(n)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        ref = np.einsum("bhqc,bhcd->bhqd", p / p.sum(axis=-1, keepdims=True), v_bar)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_rows_are_convex(self):
        q, k, _ = self._qkv(14)
        v = mx.ones((self.B, self.H, self.Sk, 3))
        out = centroid_compensated_attention(
            q,
            k,
            v,
            q_labels=_balanced_labels(self.Sq, self.Cq, 15),
            k_labels=_balanced_labels(self.Sk, self.Ck, 16),
            block_mask=_random_block_mask((self.Cq, self.Ck), 17),
        )
        assert mx.allclose(out, mx.ones_like(out), atol=1e-5).item()

    def test_per_head_labels_match_loop(self):
        q, k, v = self._qkv(18)
        ql = mx.stack([_balanced_labels(self.Sq, self.Cq, 19 + h) for h in range(self.H)])[None]
        kl = mx.stack([_balanced_labels(self.Sk, self.Ck, 29 + h) for h in range(self.H)])[None]
        bm = _random_block_mask((self.Cq, self.Ck), 39)
        out = centroid_compensated_attention(q, k, v, q_labels=ql, k_labels=kl, block_mask=bm)
        for h in range(self.H):
            sl = slice(h, h + 1)
            ref = centroid_compensated_attention(
                q[:, sl], k[:, sl], v[:, sl], q_labels=ql[0, h], k_labels=kl[0, h], block_mask=bm
            )
            assert mx.allclose(out[:, sl], ref, atol=1e-5).item()

    def test_empty_cluster_is_ignored(self):
        q, k, v = self._qkv(40)
        ql = _balanced_labels(self.Sq, self.Cq, 41)
        kl = _balanced_labels(self.Sk, self.Ck, 42)  # cluster Ck (= 4) never used
        bm = _random_block_mask((self.Cq, self.Ck + 1), 43)
        bm[:, self.Ck] = NEG_INF  # the empty cluster is skipped: its centroid must not leak in
        out = centroid_compensated_attention(q, k, v, q_labels=ql, k_labels=kl, block_mask=bm)
        ref = centroid_compensated_attention(
            q, k, v, q_labels=ql, k_labels=kl, block_mask=bm[:, : self.Ck]
        )
        assert not mx.any(mx.isnan(out)).item()
        assert mx.allclose(out, ref, atol=1e-6).item()

    def test_bf16_inputs(self):
        q, k, v = self._qkv(44)
        ql = _balanced_labels(self.Sq, self.Cq, 45)
        kl = _balanced_labels(self.Sk, self.Ck, 46)
        bm = _random_block_mask((self.Cq, self.Ck), 47)
        ref = centroid_compensated_attention(q, k, v, q_labels=ql, k_labels=kl, block_mask=bm)
        q16, k16, v16 = (x.astype(mx.bfloat16) for x in (q, k, v))
        out = centroid_compensated_attention(q16, k16, v16, q_labels=ql, k_labels=kl, block_mask=bm)
        assert out.dtype == mx.bfloat16
        assert mx.allclose(out.astype(mx.float32), ref, atol=5e-2).item()

    def test_beats_hard_drop_on_clustered_keys(self):
        # 8 tight key clusters: centroids are good stand-ins for skipped keys.
        rng = np.random.default_rng(48)
        C, per, D = 8, 16, 16
        centers_k = rng.normal(size=(C, D))
        centers_v = rng.normal(size=(C, D))
        kl_np = np.repeat(np.arange(C), per)
        k = mx.array(
            (centers_k[kl_np] + 0.05 * rng.normal(size=(C * per, D)))[None, None], dtype=mx.float32
        )
        v = mx.array(
            (centers_v[kl_np] + 0.05 * rng.normal(size=(C * per, D)))[None, None], dtype=mx.float32
        )
        q = array_from_any(rng.normal(size=(1, 1, 32, D)), dtype=mx.float32)
        ql = mx.zeros((32,), dtype=mx.int32)
        bm = mx.array([[0.0, 0.0, 0.0, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF]])
        kl = array_from_any(kl_np.astype(np.int32))
        dense = _dense(q, k, v, 1.0 / math.sqrt(D))
        comp = centroid_compensated_attention(q, k, v, q_labels=ql, k_labels=kl, block_mask=bm)
        drop = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(D), mask=mx.take(bm[0], kl)[None]
        )
        err_comp = item_float(mx.linalg.norm(comp - dense))
        err_drop = item_float(mx.linalg.norm(drop - dense))
        assert err_comp < 0.2 * err_drop

    def test_sta_recipe_matches_oracle(self):
        T, H, W, tile = 2, 4, 4, (1, 2, 2)
        labels = tile_labels(T, H, W, tile=tile)
        bm = sliding_tile_block_mask(2, 2, 2, tile=(1, 1, 1), window=(0, 0, 0))
        mx.random.seed(49)
        q, k, v = (mx.random.normal((1, 2, T * H * W, 8)) for _ in range(3))
        out = centroid_compensated_attention(
            q, k, v, q_labels=labels, k_labels=labels, block_mask=bm
        )
        ref = _replacement_oracle(q, k, v, labels, labels, bm, 1.0 / math.sqrt(8))
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_validation(self):
        q, k, v = self._qkv(50)
        ql = _balanced_labels(self.Sq, self.Cq, 51)
        kl = _balanced_labels(self.Sk, self.Ck, 52)
        bm = mx.zeros((self.Cq, self.Ck))

        def call(**overrides):
            args = {"q": q, "k": k, "v": v, "q_labels": ql, "k_labels": kl, "block_mask": bm}
            args.update(overrides)
            centroid_compensated_attention(
                args["q"],
                args["k"],
                args["v"],
                q_labels=args["q_labels"],
                k_labels=args["k_labels"],
                block_mask=args["block_mask"],
            )

        with pytest.raises(ValueError, match="rank 4"):
            call(q=q[0])
        with pytest.raises(ValueError, match="heads"):
            call(k=k[:, :1], v=v[:, :1])
        with pytest.raises(ValueError, match="key length"):
            call(v=v[:, :, :-1])
        with pytest.raises(ValueError, match="head dim"):
            call(k=k[..., :-1])
        with pytest.raises(ValueError, match="integer"):
            call(q_labels=ql.astype(mx.float32))
        with pytest.raises(ValueError, match="shape"):
            call(k_labels=kl[:-1])
        with pytest.raises(ValueError, match=r"\[0, 3\)"):
            call(q_labels=mx.full((self.Sq,), 3, dtype=mx.int32))
        with pytest.raises(ValueError, match="0 or -inf"):
            call(block_mask=mx.full((self.Cq, self.Ck), 0.5))
        with pytest.raises(ValueError, match="broadcast"):
            call(block_mask=mx.zeros((5, self.Cq, self.Ck)))
        with pytest.raises(ValueError, match="rank"):
            call(block_mask=mx.zeros((self.Ck,)))


class TestSelectProbeRows:
    # clusters: 0 -> positions [0, 1, 2], 1 -> [3, 4], 2 -> [5]
    labels = mx.array([0, 0, 0, 1, 1, 2], dtype=mx.int32)

    def test_round_robin_middle_out(self):
        idx, w = select_probe_rows(self.labels, 5)
        # pass 0 takes each cluster's middle member, pass 1 the next one out.
        assert idx.dtype == mx.int32
        assert idx.tolist() == [1, 3, 5, 2, 4]
        assert w.dtype == mx.float32
        assert w.tolist() == [1.5, 1.0, 1.0, 1.5, 1.0]

    def test_weights_sum_to_sequence_length(self):
        labels = tile_labels(2, 4, 4, tile=(1, 2, 2))
        for num in (8, 13, 32):
            _, w = select_probe_rows(labels, num)
            assert item_float(mx.sum(w)) == pytest.approx(32.0)

    def test_fewer_probes_than_clusters(self):
        idx, w = select_probe_rows(self.labels, 2)
        assert idx.tolist() == [1, 3]
        assert w.tolist() == [3.0, 2.0]

    def test_clusters_taken_in_label_order(self):
        idx, _ = select_probe_rows(mx.array([2, 0, 2, 0], dtype=mx.int32), 2)
        assert idx.tolist() == [1, 0]

    def test_all_rows_distinct(self):
        idx, _ = select_probe_rows(self.labels, 6)
        assert sorted(np.array(idx).tolist()) == list(range(6))

    def test_validation(self):
        with pytest.raises(ValueError, match="1D"):
            select_probe_rows(self.labels[None], 2)
        with pytest.raises(ValueError, match="integer"):
            select_probe_rows(self.labels.astype(mx.float32), 2)
        with pytest.raises(ValueError, match="non-negative"):
            select_probe_rows(mx.array([0, -1], dtype=mx.int32), 1)
        with pytest.raises(ValueError, match="num_probes"):
            select_probe_rows(self.labels, 0)
        with pytest.raises(ValueError, match="num_probes"):
            select_probe_rows(self.labels, 7)


class TestProbeResidualCorrection:
    B, H, S, Dv = 1, 2, 64, 8

    def _rows(self, seed: int) -> mx.array:
        rng = np.random.default_rng(seed)
        return array_from_any(rng.normal(size=(self.B, self.H, self.S, self.Dv)).astype(np.float32))

    def test_exact_sparse_is_unchanged(self):
        dense = self._rows(0)
        idx = mx.arange(0, self.S, 4)
        out = probe_residual_correction(dense, dense[:, :, idx], idx, rank=4)
        assert mx.allclose(out, dense, atol=1e-5).item()

    def test_probe_rows_are_exact(self):
        o_sparse, o_probe_full = self._rows(1), self._rows(2)
        idx = mx.array([3, 17, 40, 41, 60, 5, 22, 9, 33], dtype=mx.int32)
        out = probe_residual_correction(o_sparse, o_probe_full[:, :, idx], idx, rank=4)
        assert mx.array_equal(out[:, :, idx], o_probe_full[:, :, idx]).item()

    def test_recovers_planted_low_rank_residual(self):
        # dense = sparse + sparse @ A + c with rank(A) = 2: an affine residual the
        # ridge + rank-2 projection must recover on every row, not just the probes.
        rng = np.random.default_rng(3)
        o_sparse = self._rows(4)
        A = array_from_any(
            (rng.normal(size=(self.Dv, 2)) @ rng.normal(size=(2, self.Dv))).astype(np.float32)
        )
        c = array_from_any(rng.normal(size=(self.Dv,)).astype(np.float32))
        dense = o_sparse + o_sparse @ A + c
        idx = mx.arange(0, self.S, 2)
        out = probe_residual_correction(o_sparse, dense[:, :, idx], idx, rank=2, ridge=1e-6)
        assert mx.allclose(out, dense, atol=1e-3).item()

    def test_reduces_error_on_sta_drop(self):
        T, H, W, D = 2, 8, 8, 16
        S = T * H * W
        labels = tile_labels(T, H, W, tile=(1, 4, 4))
        bm = sliding_tile_block_mask(2, 2, 2, tile=(1, 1, 1), window=(0, 0, 0))[0, 0]
        tok_mask = mx.take(mx.take(bm, labels, axis=0), labels, axis=1)
        rng = np.random.default_rng(0)
        q, k, v = (
            array_from_any(rng.normal(size=(1, 2, S, D)).astype(np.float32)) for _ in range(3)
        )
        scale = 1.0 / math.sqrt(D)
        dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        sparse = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=tok_mask)
        idx, w = select_probe_rows(labels, 32)
        o_probe = mx.fast.scaled_dot_product_attention(q[:, :, idx], k, v, scale=scale)
        out = probe_residual_correction(sparse, o_probe, idx, rank=4, weights=w)
        rest = array_from_any(np.setdiff1d(np.arange(S), np.array(idx)).astype(np.int32))
        err_before = item_float(mx.linalg.norm(mx.take(sparse - dense, rest, axis=2)))
        err_after = item_float(mx.linalg.norm(mx.take(out - dense, rest, axis=2)))
        assert err_after < 0.9 * err_before

    def test_preserves_dtype(self):
        dense = self._rows(5).astype(mx.float16)
        idx = mx.arange(0, self.S, 4)
        out = probe_residual_correction(dense, dense[:, :, idx], idx, rank=4)
        assert out.dtype == mx.float16

    def test_validation(self):
        o = self._rows(6)
        idx = mx.arange(0, 16)
        op = o[:, :, idx]
        with pytest.raises(ValueError, match="rank 4"):
            probe_residual_correction(o[0], op, idx)
        with pytest.raises(ValueError, match="1D"):
            probe_residual_correction(o, op, idx[None])
        with pytest.raises(ValueError, match="integer"):
            probe_residual_correction(o, op, idx.astype(mx.float32))
        with pytest.raises(ValueError, match="o_probe"):
            probe_residual_correction(o, op[:, :, :-1], idx)
        with pytest.raises(ValueError, match=r"\[0, 64\)"):
            probe_residual_correction(o, op, idx + 60)
        with pytest.raises(ValueError, match="distinct"):
            probe_residual_correction(o, op, mx.zeros((16,), dtype=mx.int32))
        with pytest.raises(ValueError, match="ridge"):
            probe_residual_correction(o, op, idx, ridge=0.0)
        with pytest.raises(ValueError, match="rank"):
            probe_residual_correction(o, op, idx, rank=0)
        with pytest.raises(ValueError, match="rank"):
            probe_residual_correction(o, op, idx, rank=self.Dv + 1)
        with pytest.raises(ValueError, match="weights"):
            probe_residual_correction(o, op, idx, rank=4, weights=mx.ones((15,)))
        with pytest.raises(ValueError, match="weights"):
            probe_residual_correction(o, op, idx, rank=4, weights=-mx.ones((16,)))
        with pytest.raises(ValueError, match="weights"):
            probe_residual_correction(o, op, idx, rank=4, weights=mx.zeros((16,)))
