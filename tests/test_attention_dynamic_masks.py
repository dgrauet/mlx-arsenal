"""Tests for mlx_arsenal.attention.dynamic_masks."""

import math
import re
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.attention import (
    antidiagonal_block_scores,
    block_self_similarity,
    centroid_compensated_attention,
    minmax_block_scores,
    radius_bounded_block_scores,
    relative_block_mask,
    select_tiling,
    tile_labels,
    top_k_block_mask,
    top_p_block_mask,
)


def _qk(B: int, H: int, Nq: int, Nk: int, D: int, seed: int) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    q = array_from_any(rng.normal(size=(B, H, Nq, D)).astype(np.float32))
    k = array_from_any(rng.normal(size=(B, H, Nk, D)).astype(np.float32))
    return q, k


def _top_p_reference(scores: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """XAttention non-causal rule: keep a block while the mass ranked before it < tau * total."""
    B, H, Cq, Ck = scores.shape
    keep = np.zeros(scores.shape, dtype=bool)
    for b in range(B):
        for h in range(H):
            for i in range(Cq):
                row = scores[b, h, i].astype(np.float64)
                order = np.argsort(-row, kind="stable")
                before = np.concatenate([[0.0], np.cumsum(row[order])[:-1]])
                kept = order[before < tau[h] * row.sum()]
                keep[b, h, i, kept if kept.size else order[:1]] = True
    return keep


def _brute_force_scores(q, k, block_size: int, stride: int, scale: float) -> np.ndarray:
    """Anti-diagonal sums of every stride x stride sub-block, softmax, tile sums."""
    qn, kn = np.array(q, dtype=np.float64), np.array(k, dtype=np.float64)
    B, H, Nq, _ = qn.shape
    Nk = kn.shape[2]
    rq, rk = Nq // stride, Nk // stride
    tile = block_size // stride
    out = np.zeros((B, H, Nq // block_size, Nk // block_size))
    for b in range(B):
        for h in range(H):
            logits = np.zeros((rq, rk))
            for i in range(rq):
                for j in range(rk):
                    s = sum(
                        qn[b, h, i * stride + stride - 1 - r] @ kn[b, h, j * stride + r]
                        for r in range(stride)
                    )
                    logits[i, j] = s * scale / stride
            p = np.exp(logits - logits.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            sums = p.reshape(rq // tile, tile, rk // tile, tile).sum(axis=(1, 3))
            out[b, h] = sums / tile
    return out


class TestAntidiagonalBlockScores:
    @pytest.mark.parametrize(("Nq", "Nk", "block_size", "stride"), [(32, 32, 8, 4), (16, 48, 8, 2)])
    def test_matches_brute_force(self, Nq, Nk, block_size, stride):
        q, k = _qk(1, 2, Nq, Nk, 8, 0)
        out = antidiagonal_block_scores(q, k, block_size=block_size, stride=stride)
        ref = _brute_force_scores(q, k, block_size, stride, 1.0 / math.sqrt(8))
        assert out.dtype == mx.float32
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_rows_sum_to_one(self):
        q, k = _qk(2, 3, 64, 64, 16, 1)
        out = antidiagonal_block_scores(q, k, block_size=16, stride=4)
        assert out.shape == (2, 3, 4, 4)
        assert mx.allclose(mx.sum(out, axis=-1), mx.ones((2, 3, 4)), atol=1e-5).item()

    @pytest.mark.parametrize("stride", [1, 8])
    def test_stride_extremes(self, stride):
        q, k = _qk(1, 1, 16, 16, 4, 2)
        out = antidiagonal_block_scores(q, k, block_size=8, stride=stride, scale=0.7)
        ref = _brute_force_scores(q, k, 8, stride, 0.7)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_bf16_inputs(self):
        q, k = _qk(1, 2, 32, 32, 8, 3)
        out = antidiagonal_block_scores(
            q.astype(mx.bfloat16), k.astype(mx.bfloat16), block_size=8, stride=4
        )
        ref = antidiagonal_block_scores(q, k, block_size=8, stride=4)
        assert out.dtype == mx.float32
        assert mx.all(mx.isfinite(out)).item()
        assert mx.allclose(out, ref, atol=2e-2).item()

    def test_validation(self):
        q, k = _qk(1, 2, 32, 32, 8, 4)
        with pytest.raises(ValueError, match="rank 4"):
            antidiagonal_block_scores(q[0], k, block_size=8, stride=4)
        with pytest.raises(ValueError, match="heads"):
            antidiagonal_block_scores(q, k[:, :1], block_size=8, stride=4)
        with pytest.raises(ValueError, match="head dim"):
            antidiagonal_block_scores(q, k[..., :4], block_size=8, stride=4)
        with pytest.raises(ValueError, match="multiple of block_size"):
            antidiagonal_block_scores(q[:, :, :30], k, block_size=8, stride=4)
        with pytest.raises(ValueError, match="multiple of block_size"):
            antidiagonal_block_scores(q, k[:, :, :30], block_size=8, stride=4)
        with pytest.raises(ValueError, match="multiple of stride"):
            antidiagonal_block_scores(q, k, block_size=8, stride=3)
        with pytest.raises(ValueError, match="block_size"):
            antidiagonal_block_scores(q, k, block_size=0, stride=1)
        with pytest.raises(ValueError, match="stride"):
            antidiagonal_block_scores(q, k, block_size=8, stride=0)


class TestTopPBlockMask:
    def _scores(self, seed: int, shape=(2, 3, 4, 6)) -> mx.array:
        rng = np.random.default_rng(seed)
        raw = rng.random(shape) ** 3  # skewed, like attention mass
        return array_from_any((raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32))

    @pytest.mark.parametrize("tau", [0.3, 0.5, 0.9, 0.99])
    def test_matches_reference(self, tau):
        scores = self._scores(10)
        out = top_p_block_mask(scores, tau)
        ref = _top_p_reference(np.array(scores), np.full(3, tau))
        assert out.dtype == mx.float32
        assert np.array_equal(np.array(out) == 0.0, ref)
        assert np.isneginf(np.array(out)[~ref]).all()

    def test_per_head_thresholds(self):
        scores = self._scores(11)
        tau = mx.array([0.2, 0.6, 0.95])
        out = top_p_block_mask(scores, tau)
        ref = _top_p_reference(np.array(scores), np.array(tau))
        assert np.array_equal(np.array(out) == 0.0, ref)
        kept = np.array(mx.sum(out == 0.0, axis=(0, 2, 3)))
        assert kept[0] < kept[1] < kept[2]

    def test_kept_mass_reaches_threshold(self):
        scores = self._scores(12)
        out = top_p_block_mask(scores, 0.8)
        kept_mass = mx.sum(mx.where(out == 0.0, scores, 0.0), axis=-1)
        assert mx.all(kept_mass >= 0.8 - 1e-6).item()

    def test_tie_policy_holds_at_scale(self):
        # Equal scores keep the lower key block first (stable argsort), also on long rows.
        Ck = 4096
        scores = mx.full((1, 1, 1, Ck), 1.0 / Ck)
        keep = np.array(top_p_block_mask(scores, 0.1))[0, 0, 0] == 0.0
        n = int(np.ceil(0.1 * Ck))  # blocks needed to reach 10 % of the mass
        assert np.flatnonzero(keep).tolist() == list(range(n))

    def test_zero_row_keeps_one(self):
        scores = mx.zeros((1, 1, 2, 4))
        out = top_p_block_mask(scores, 0.9)
        assert not mx.any(mx.isnan(out)).item()
        assert np.array(mx.sum(out == 0.0, axis=-1)).tolist() == [[[1, 1]]]

    def test_block_zero_is_not_forced(self):
        # XAttention's scatter always re-marks key block 0; we keep only the top-p set.
        scores = mx.array([[[[0.01, 0.9, 0.09]]]])
        assert (np.array(top_p_block_mask(scores, 0.5)) == 0.0).tolist() == [
            [[[False, True, False]]]
        ]

    def test_full_threshold_with_compensation_equals_dense(self):
        q, k = _qk(1, 2, 32, 32, 8, 13)
        v = _qk(1, 2, 32, 32, 8, 14)[0]
        bm = top_p_block_mask(antidiagonal_block_scores(q, k, block_size=8, stride=4), 1.0)
        labels = mx.arange(32) // 8
        out = centroid_compensated_attention(
            q, k, v, q_labels=labels, k_labels=labels, block_mask=bm
        )
        dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(8))
        assert mx.allclose(out, dense, atol=1e-5).item()

    def test_validation(self):
        scores = self._scores(15)
        with pytest.raises(ValueError, match="rank 4"):
            top_p_block_mask(scores[0], 0.9)
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="threshold"):
                top_p_block_mask(scores, bad)
        with pytest.raises(ValueError, match="threshold"):
            top_p_block_mask(scores, mx.array([0.9, 0.9]))
        with pytest.raises(ValueError, match="threshold"):
            top_p_block_mask(scores, mx.array([0.9, 0.0, 0.9]))


def _spade_block_summarize(x: np.ndarray, block_size: int) -> np.ndarray:
    """Transcription of SPADE's torch `_block_summarize` (mean off-diagonal cosine)."""
    B, H, N, D = x.shape
    t = x.astype(np.float64).reshape(B, H, N // block_size, block_size, D)
    norm = np.linalg.norm(t, axis=-1, keepdims=True)
    tn = t / (norm + 1e-8)
    cos = tn @ np.swapaxes(tn, -1, -2)
    valid = norm[..., 0] > 1e-8
    pair = valid[..., :, None] & valid[..., None, :]
    final = pair & ~np.eye(block_size, dtype=bool)
    return np.where(final, cos, 0).sum(axis=(-1, -2)) / (final.sum(axis=(-1, -2)) + 1e-8)


class TestBlockSelfSimilarity:
    def test_matches_spade_reference(self):
        q, _ = _qk(2, 3, 64, 64, 8, 30)
        out = block_self_similarity(q, block_size=16)
        assert out.shape == (2, 3, 4)
        assert out.dtype == mx.float32
        np.testing.assert_allclose(
            np.array(out), _spade_block_summarize(np.array(q), 16), atol=1e-5
        )

    def test_zero_norm_tokens_excluded(self):
        q, _ = _qk(1, 2, 32, 32, 8, 31)
        qn = np.array(q)
        qn[:, :, ::3] = 0.0  # padding tokens
        out = block_self_similarity(array_from_any(qn), block_size=8)
        np.testing.assert_allclose(np.array(out), _spade_block_summarize(qn, 8), atol=1e-5)

    def test_single_valid_token_block(self):
        x = np.zeros((1, 1, 8, 4), dtype=np.float32)
        x[0, 0, 0] = 1.0  # block 0: one valid token; block 1: none
        out = block_self_similarity(array_from_any(x), block_size=4)
        assert np.array(out).tolist() == [[[0.0, 0.0]]]

    def test_identical_tokens_have_similarity_one(self):
        x = mx.broadcast_to(mx.array([0.3, -1.0, 2.0]), (1, 1, 8, 3))
        assert mx.allclose(
            block_self_similarity(x, block_size=4), mx.ones((1, 1, 2)), atol=1e-6
        ).item()

    def test_bf16_inputs(self):
        q, _ = _qk(1, 2, 32, 32, 8, 32)
        out = block_self_similarity(q.astype(mx.bfloat16), block_size=8)
        assert out.dtype == mx.float32
        assert mx.allclose(out, block_self_similarity(q, block_size=8), atol=2e-2).item()

    def test_validation(self):
        q, _ = _qk(1, 2, 32, 32, 8, 33)
        with pytest.raises(ValueError, match="rank 4"):
            block_self_similarity(q[0], block_size=8)
        with pytest.raises(ValueError, match="multiple of block_size"):
            block_self_similarity(q, block_size=5)
        with pytest.raises(ValueError, match="block_size"):
            block_self_similarity(q, block_size=0)


class TestSelectTiling:
    T, Hh, W = 2, 4, 8  # 64 tokens
    SPATIAL, TEMPORAL = (1, 4, 4), (2, 1, 8)

    def _coherent(self, tile: tuple[int, int, int], seed: int) -> np.ndarray:
        """(N, D) queries sharing one random direction per tile of `tile`, plus small noise."""
        rng = np.random.default_rng(seed)
        labels = np.array(tile_labels(self.T, self.Hh, self.W, tile=tile))
        dirs = rng.normal(size=(labels.max() + 1, 16))
        return (dirs[labels] + 0.05 * rng.normal(size=(labels.size, 16))).astype(np.float32)

    def test_picks_the_coherent_tiling_per_head(self):
        q = np.stack([self._coherent(self.SPATIAL, 1), self._coherent(self.TEMPORAL, 2)])[None]
        out = select_tiling(
            array_from_any(q), (self.T, self.Hh, self.W), [self.SPATIAL, self.TEMPORAL]
        )
        assert out.dtype == mx.int32
        assert out.tolist() == [[0, 1]]

    def test_per_head_choice(self):
        heads = [self._coherent(t, s) for t, s in ((self.TEMPORAL, 3), (self.SPATIAL, 4))]
        q = np.stack([np.stack(heads), np.stack(heads[::-1])])  # (B=2, H=2, N, D)
        out = select_tiling(
            array_from_any(q), (self.T, self.Hh, self.W), [self.SPATIAL, self.TEMPORAL]
        )
        assert out.tolist() == [[1, 0], [0, 1]]

    def test_ties_pick_the_first_candidate(self):
        q = mx.broadcast_to(mx.array([1.0, 2.0]), (1, 1, 64, 2))
        out = select_tiling(q, (self.T, self.Hh, self.W), [self.TEMPORAL, self.SPATIAL])
        assert out.tolist() == [[0]]

    def test_validation(self):
        q = mx.zeros((1, 1, 64, 4))
        grid = (self.T, self.Hh, self.W)
        with pytest.raises(ValueError, match="grid"):
            select_tiling(q, (2, 4, 4), [self.SPATIAL])
        with pytest.raises(ValueError, match="tiles"):
            select_tiling(q, grid, [])
        with pytest.raises(ValueError, match="divisible"):
            select_tiling(q, grid, [(1, 3, 4)])
        with pytest.raises(ValueError, match="rank 4"):
            select_tiling(q[0], grid, [self.SPATIAL])


def _spade_minmax(q: np.ndarray, k: np.ndarray, block_size: int) -> np.ndarray:
    """Transcription of SPADE's `estimate_func_minmax` on contiguous blocks."""
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    qb = q.astype(np.float64).reshape(B, H, Nq // block_size, block_size, D)
    kb = k.astype(np.float64).reshape(B, H, Nk // block_size, block_size, D)
    qsum = qb.max(axis=3) + qb.min(axis=3)
    return np.maximum(
        qsum @ np.swapaxes(kb.max(axis=3), -1, -2), qsum @ np.swapaxes(kb.min(axis=3), -1, -2)
    )


class TestMinmaxBlockScores:
    def test_matches_spade_reference(self):
        q, k = _qk(2, 3, 32, 48, 8, 40)
        out = minmax_block_scores(q, k, block_size=8)
        assert out.shape == (2, 3, 4, 6)
        assert out.dtype == mx.float32
        np.testing.assert_allclose(
            np.array(out), _spade_minmax(np.array(q), np.array(k), 8), rtol=1e-5, atol=1e-5
        )

    def test_bf16_inputs(self):
        q, k = _qk(1, 2, 32, 32, 8, 41)
        out = minmax_block_scores(q.astype(mx.bfloat16), k.astype(mx.bfloat16), block_size=8)
        assert out.dtype == mx.float32
        assert mx.allclose(
            out, minmax_block_scores(q, k, block_size=8), rtol=2e-2, atol=5e-2
        ).item()

    def test_validation(self):
        q, k = _qk(1, 2, 32, 32, 8, 42)
        with pytest.raises(ValueError, match="rank 4"):
            minmax_block_scores(q[0], k, block_size=8)
        with pytest.raises(ValueError, match="heads"):
            minmax_block_scores(q, k[:, :1], block_size=8)
        with pytest.raises(ValueError, match="head dim"):
            minmax_block_scores(q, k[..., :4], block_size=8)
        with pytest.raises(ValueError, match="multiple of block_size"):
            minmax_block_scores(q, k, block_size=5)


class TestTopKBlockMask:
    def test_matches_topk_reference(self):
        rng = np.random.default_rng(43)
        scores = rng.normal(size=(2, 3, 4, 10)).astype(np.float32)
        out = np.array(top_k_block_mask(array_from_any(scores), 3)) == 0.0
        ref = np.zeros_like(out)
        top = np.argsort(-scores, axis=-1, kind="stable")[..., :3]
        np.put_along_axis(ref, top, True, axis=-1)
        assert np.array_equal(out, ref)
        assert np.isneginf(np.array(top_k_block_mask(array_from_any(scores), 3))[~ref]).all()

    def test_k_clipped(self):
        scores = mx.array([[[[0.1, 0.5, 0.2]]]])
        assert (np.array(top_k_block_mask(scores, 10)) == 0.0).all()

    def test_ties_keep_lower_index(self):
        scores = mx.array([[[[1.0, 2.0, 2.0, 2.0]]]])
        assert (np.array(top_k_block_mask(scores, 2)) == 0.0).tolist() == [
            [[[False, True, True, False]]]
        ]

    def test_validation(self):
        with pytest.raises(ValueError, match="rank 4"):
            top_k_block_mask(mx.zeros((2, 3)), 1)
        with pytest.raises(ValueError, match="k"):
            top_k_block_mask(mx.zeros((1, 1, 2, 3)), 0)


def _load_spade_recipe() -> Any:
    note = Path(__file__).parents[1] / "docs" / "research" / "adaptive-tiling.md"
    match = re.search(r"<!-- spade-recipe -->\s*```python\n(.*?)```", note.read_text(), re.S)
    assert match, "SPADE recipe block not found in the research note"
    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)
    return namespace["spade_attention"]


class TestSpadeRecipe:
    GRID, TILES = (2, 4, 8), [(1, 4, 4), (2, 1, 8)]

    def test_full_budget_equals_dense_attention(self):
        # budget 1.0 keeps every block: the per-group permute / inverse-permute
        # must then reproduce dense attention exactly.
        spade = _load_spade_recipe()
        q, k = _qk(1, 4, 64, 64, 8, 50)
        v = _qk(1, 4, 64, 64, 8, 51)[0]
        out = spade(q, k, v, self.GRID, self.TILES, budget=1.0)
        dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(8))
        assert mx.allclose(out, dense, atol=1e-5).item()

    def test_sparse_budget_runs_with_mixed_tilings(self):
        spade = _load_spade_recipe()
        rng = np.random.default_rng(52)
        heads = []
        for tile in self.TILES:  # one spatially and one temporally coherent head
            labels = np.array(tile_labels(*self.GRID, tile=tile))
            heads.append(
                rng.normal(size=(labels.max() + 1, 8))[labels] + 0.05 * rng.normal(size=(64, 8))
            )
        q = array_from_any(np.stack(heads)[None].astype(np.float32))
        assert select_tiling(q, self.GRID, self.TILES).tolist() == [[0, 1]]
        k = array_from_any(rng.normal(size=(1, 2, 64, 8)).astype(np.float32))
        out = spade(q, k, k, self.GRID, self.TILES, budget=0.25)
        assert out.shape == (1, 2, 64, 8)
        assert mx.all(mx.isfinite(out)).item()
        # Independent per-head reference (no grouping): each head with its own
        # tiling, min/max estimate, top-k budget and compensated attention.
        choice = cast(list[int], select_tiling(q, self.GRID, self.TILES)[0].tolist())
        for h, pick in enumerate(choice):
            tile = self.TILES[pick]
            order = mx.argsort(tile_labels(*self.GRID, tile=tile))
            qh, kh = (mx.take(x[:, h : h + 1], order, axis=2) for x in (q, k))
            block = tile[0] * tile[1] * tile[2]
            scores = minmax_block_scores(qh, kh, block_size=block)
            mask = top_k_block_mask(scores, max(1, int(0.25 * scores.shape[-1])))
            labels = mx.arange(64) // block
            ref = centroid_compensated_attention(
                qh, kh, kh, q_labels=labels, k_labels=labels, block_mask=mask
            )
            ref = mx.take(ref, mx.argsort(order), axis=2)
            assert mx.allclose(out[:, h : h + 1], ref, atol=1e-5).item()
            # the sparse mask actually changes the result vs dense attention
            dense = mx.fast.scaled_dot_product_attention(
                q[:, h : h + 1], k[:, h : h + 1], k[:, h : h + 1], scale=1.0 / math.sqrt(8)
            )
            assert not mx.allclose(out[:, h : h + 1], dense, atol=1e-3).item()

    def test_recipe_rejects_batches(self):
        # select_tiling decides per (batch, head); the recipe is batch-1 and must
        # not silently apply batch 0's choice to every batch.
        spade = _load_spade_recipe()
        q, k = _qk(2, 2, 64, 64, 8, 53)
        with pytest.raises(ValueError, match="batch"):
            spade(q, k, k, self.GRID, self.TILES)


def _rbs_reference(q, k, bs, low=0.5, high=0.9):
    """Transcription of RBS-Attention Algorithm 1 scores (float64), as log scores."""
    B, H, Nq, D = q.shape
    Ck = k.shape[2] // bs
    base = np.zeros((B, H, Nq // bs, Ck))
    rescue = np.zeros_like(base)
    for b in range(B):
        for h in range(H):
            kb = k[b, h].astype(np.float64).reshape(Ck, bs, D)
            c = kb.mean(1)
            r = np.linalg.norm(kb - c[:, None], axis=-1).max(1)
            r_lo, r_hi = np.quantile(r, low), np.quantile(r, high)
            if r_hi > r_lo:
                beta = np.clip((r - r_lo) / (r_hi - r_lo), 0, 1)
            else:
                beta = (r > r_lo).astype(np.float64)
            qq = q[b, h].astype(np.float64)
            lb = qq @ c.T / np.sqrt(D)
            lr = (qq @ c.T + np.linalg.norm(qq, axis=-1, keepdims=True) * (r * beta)) / np.sqrt(D)
            for i in range(Nq // bs):
                for out, lg in ((base, lb), (rescue, lr)):
                    blk = lg[i * bs : (i + 1) * bs]
                    m = blk.max(0)
                    out[b, h, i] = m + np.log(np.exp(blk - m).sum(0))
    return base, rescue


class TestRadiusBoundedBlockScores:
    def _qk(self, seed=0, B=2, H=3, N=64, D=16):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((B, H, N, D)).astype(np.float32) for _ in range(2))

    @pytest.mark.parametrize(("low", "high"), [(0.5, 0.9), (0.25, 0.75)])
    def test_matches_algorithm_1(self, low, high):
        q, k = self._qk()
        base, rescue = radius_bounded_block_scores(
            array_from_any(q), array_from_any(k), block_size=8, low=low, high=high
        )
        ref_b, ref_r = _rbs_reference(q, k, 8, low, high)
        np.testing.assert_allclose(np.array(base), ref_b, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.array(rescue), ref_r, atol=1e-4, rtol=1e-4)

    def test_full_rescue_bounds_the_true_block_score(self):
        # Blocks with radius >= the `high` quantile get beta = 1: their rescue
        # logit is the Cauchy-Schwarz upper bound of every key logit, so the
        # block score bounds the one built from each query's best key.
        q, k = self._qk(1, B=1, H=1)
        _, rescue = radius_bounded_block_scores(
            array_from_any(q), array_from_any(k), block_size=8, low=0.0, high=0.5
        )
        kb = k[0, 0].astype(np.float64).reshape(8, 8, 16)
        r = np.linalg.norm(kb - kb.mean(1, keepdims=True), axis=-1).max(1)
        full = r >= np.quantile(r, 0.5)
        logits = q[0, 0].astype(np.float64) @ k[0, 0].astype(np.float64).T / 4.0  # sqrt(D)
        best = logits.reshape(8, 8, 8, 8).max(axis=3)  # (Cq, tokens, Ck)
        true = np.log(np.exp(best).sum(axis=1))  # (Cq, Ck)
        got = np.array(rescue)[0, 0]
        assert full.sum() >= 4
        assert np.all(got[:, full] >= true[:, full] - 1e-4)

    def test_rescue_keeps_a_diluted_key(self):
        # Key block 3 hides one key aligned with the queries among keys that
        # cancel it on average: the centroid misses it, the radius bound does not.
        D, bs = 8, 8
        rng = np.random.default_rng(2)
        q = np.tile(np.eye(D)[0] * 4.0, (1, 1, bs, 1)).astype(np.float32)
        k = 0.05 * rng.standard_normal((1, 1, 8 * bs, D))
        k[0, 0, 3 * bs] = np.eye(D)[0] * 6.0
        k[0, 0, 3 * bs + 1 : 4 * bs, 0] = -6.0 / (bs - 1)
        k[0, 0, 5 * bs : 6 * bs, 0] += 1.0  # a decoy: tight block, moderately aligned
        k = k.astype(np.float32)
        # The hidden key is the best match of every query (logit 4·6 vs 4·1).
        assert (q[0, 0, 0] @ k[0, 0].T).argmax() == 3 * bs
        base, rescue = radius_bounded_block_scores(
            array_from_any(q), array_from_any(k), block_size=bs
        )
        keep_b = np.array(relative_block_mask(base, 0.5))[0, 0, 0]
        keep_r = np.array(relative_block_mask(rescue, 0.5))[0, 0, 0]
        assert keep_b[3] == -np.inf
        assert keep_r[3] == 0.0

    def test_bf16_inputs(self):
        q, k = self._qk(3)
        b16, r16 = radius_bounded_block_scores(
            array_from_any(q).astype(mx.bfloat16),
            array_from_any(k).astype(mx.bfloat16),
            block_size=8,
        )
        assert b16.dtype == mx.float32 and r16.dtype == mx.float32

    @pytest.mark.parametrize(
        "kw",
        [
            {"block_size": 0},
            {"block_size": 7},
            {"block_size": 8, "low": 0.9, "high": 0.5},
            {"block_size": 8, "low": -0.1},
            {"block_size": 8, "high": 1.1},
        ],
    )
    def test_validation(self, kw):
        q, k = self._qk(4)
        with pytest.raises(ValueError):
            radius_bounded_block_scores(array_from_any(q), array_from_any(k), **kw)


class TestRelativeBlockMask:
    def test_keeps_scores_within_alpha_of_the_max(self):
        rng = np.random.default_rng(5)
        logs = rng.standard_normal((2, 3, 4, 10)).astype(np.float32)
        out = np.array(relative_block_mask(array_from_any(logs), 0.22))
        keep = logs >= logs.max(-1, keepdims=True) + np.log(0.22)
        np.testing.assert_array_equal(out == 0.0, keep)
        assert np.all(out[~keep] == -np.inf)

    def test_alpha_one_keeps_the_max(self):
        logs = mx.array([[[[0.0, 2.0, 1.0]]]])
        assert np.array(relative_block_mask(logs, 1.0)).tolist() == [[[[-np.inf, 0.0, -np.inf]]]]

    @pytest.mark.parametrize("alpha", [0.0, -0.5, 1.5])
    def test_invalid_alpha(self, alpha):
        with pytest.raises(ValueError):
            relative_block_mask(mx.zeros((1, 1, 2, 3)), alpha)
