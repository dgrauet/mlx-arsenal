"""Tests for mlx_arsenal.diffusion.masked_decode."""

import math
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any, item_float
from mlx_arsenal.diffusion import (
    block_ranges,
    entropy_bound_transfer,
    factor_transfer,
    threshold_transfer,
    token_stats,
    topk_transfer,
    transfer_schedule,
)


def _np_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _fastdllm_transfer(
    conf: np.ndarray, mask: np.ndarray, k: np.ndarray | None, threshold: float | None
) -> np.ndarray:
    """Transcription of Fast-dLLM v1 `get_transfer_index` selection (float64)."""
    confidence = np.where(mask, conf.astype(np.float64), -np.inf)
    num = mask.sum(axis=1) if threshold is not None else k
    assert num is not None
    out = np.zeros_like(mask)
    for j in range(conf.shape[0]):
        if num[j] == 0:
            continue
        select = np.argsort(-confidence[j], kind="stable")[: num[j]]
        out[j, select] = True
        if threshold is not None:
            for idx in select[1:]:
                if confidence[j, idx] < threshold:
                    out[j, idx] = False
    return out


def _fastdllm_factor(
    conf: np.ndarray, mask: np.ndarray, factor: float, *, quirk: bool
) -> np.ndarray:
    """Transcription of Fast-dLLM v1 `get_transfer_index_dynamic` selection.

    With `quirk=True` it keeps the reference's off-by-one: when only the last
    sorted candidate fails, every candidate is selected.
    """
    confidence = np.where(mask, conf.astype(np.float64), -np.inf)
    out = np.zeros_like(mask)
    for j in range(conf.shape[0]):
        n_cand = int(mask[j].sum())
        if n_cand == 0:
            continue
        threshs = [1 - factor / (n + 1) for n in range(1, n_cand + 1)]
        threshs[0] = -1
        sorted_c = np.sort(confidence[j][mask[j]])[::-1]
        top_i = 0
        broke = False
        for top_i in range(n_cand):
            if sorted_c[top_i] < threshs[top_i]:
                broke = True
                break
        if quirk:
            if top_i == 0 or top_i == n_cand - 1:
                top_i += 1
        elif not broke:
            top_i = n_cand
        select = np.argsort(-confidence[j], kind="stable")[:top_i]
        out[j, select] = True
    return out


def _eb_accept(entropy: np.ndarray, mask: np.ndarray, bound: float) -> np.ndarray:
    """EB-Sampler acceptance (HF DiffusionGemma): keep while cumsum - current <= bound."""
    out = np.zeros_like(mask)
    for j in range(entropy.shape[0]):
        idx = np.flatnonzero(mask[j])
        if idx.size == 0:
            continue
        order = idx[np.argsort(entropy[j, idx].astype(np.float64), kind="stable")]
        h = entropy[j, order].astype(np.float64)
        accept = (np.cumsum(h) - h) <= bound
        out[j, order[accept]] = True
    return out


def _random_case(B: int, L: int, seed: int, p_mask: float = 0.6):
    rng = np.random.default_rng(seed)
    conf = rng.random((B, L)).astype(np.float32)
    mask = rng.random((B, L)) < p_mask
    return conf, mask


def _random_logits(shape: tuple[int, ...], seed: int, scale: float = 3.0) -> np.ndarray:
    return (scale * np.random.default_rng(seed).normal(size=shape)).astype(np.float32)


class TestTokenStats:
    def test_greedy_matches_float64_reference(self):
        logits = _random_logits((2, 5, 11), 0)
        stats = token_stats(array_from_any(logits))
        p = _np_softmax(logits.astype(np.float64))
        x0 = p.argmax(axis=-1)
        assert stats.x0.dtype == mx.int32
        assert np.array(stats.x0).tolist() == x0.tolist()
        np.testing.assert_allclose(
            np.array(stats.prob), np.take_along_axis(p, x0[..., None], -1)[..., 0], atol=1e-6
        )
        entropy = -(p * np.log(p)).sum(axis=-1)
        np.testing.assert_allclose(np.array(stats.entropy), entropy, atol=1e-5)
        assert stats.prob.dtype == stats.entropy.dtype == mx.float32

    def test_suppressed_ids_never_proposed(self):
        logits = _random_logits((1, 4, 9), 1)
        logits[..., 3] = 50.0  # id 3 would win everywhere
        stats = token_stats(array_from_any(logits), suppress_ids=[3])
        kept = np.delete(logits.astype(np.float64), 3, axis=-1)
        p = _np_softmax(kept)
        ref_x0 = p.argmax(axis=-1)
        ref_x0 = ref_x0 + (ref_x0 >= 3)  # map back to full-vocabulary ids
        assert np.array(stats.x0).tolist() == ref_x0.tolist()
        np.testing.assert_allclose(np.array(stats.prob), p.max(axis=-1), atol=1e-6)

    def test_neg_inf_logits_are_nan_free(self):
        logits = _random_logits((2, 3, 6), 2)
        logits[..., :4] = -np.inf  # only 2 live tokens per position
        for dtype in (mx.float32, mx.bfloat16):
            stats = token_stats(array_from_any(logits).astype(dtype))
            assert mx.all(mx.isfinite(stats.prob)).item()
            assert mx.all(mx.isfinite(stats.entropy)).item()
            assert item_float(mx.max(stats.entropy)) <= math.log(2) + 1e-5
            assert item_float(mx.min(mx.array(stats.x0))) >= 4

    def test_bf16_logits(self):
        logits = array_from_any(_random_logits((2, 8, 32), 3)).astype(mx.bfloat16)
        stats = token_stats(logits)
        ref = token_stats(logits.astype(mx.float32))
        assert mx.array_equal(stats.x0, ref.x0).item()
        assert mx.allclose(stats.entropy, ref.entropy, atol=1e-5).item()

    def test_sampling_is_deterministic_for_a_key(self):
        logits = array_from_any(_random_logits((2, 16, 50), 4, scale=1.0))
        a = token_stats(logits, temperature=1.0, key=mx.random.key(7))
        b = token_stats(logits, temperature=1.0, key=mx.random.key(7))
        c = token_stats(logits, temperature=1.0, key=mx.random.key(8))
        assert mx.array_equal(a.x0, b.x0).item()
        assert not mx.array_equal(a.x0, c.x0).item()

    @pytest.mark.parametrize(("temperature", "expected"), [(1.0, 0.8), (0.5, 0.64 / 0.68)])
    def test_sampling_follows_tempered_softmax(self, temperature, expected):
        n = 20000
        logits = mx.broadcast_to(mx.log(mx.array([0.8, 0.2])), (1, n, 2))
        stats = token_stats(logits, temperature=temperature, key=mx.random.key(0))
        freq = item_float(mx.mean((stats.x0 == 0).astype(mx.float32)))
        assert freq == pytest.approx(expected, abs=0.015)

    def test_confidence_uses_unnoised_logits(self):
        logits = _random_logits((1, 64, 5), 5, scale=1.0)
        stats = token_stats(array_from_any(logits), temperature=2.0, key=mx.random.key(1))
        p = _np_softmax(logits.astype(np.float64))
        x0 = np.array(stats.x0)
        np.testing.assert_allclose(
            np.array(stats.prob), np.take_along_axis(p, x0[..., None], -1)[..., 0], atol=1e-6
        )

    def test_greedy_ignores_key(self):
        logits = array_from_any(_random_logits((1, 6, 7), 6))
        a = token_stats(logits)
        b = token_stats(logits, key=mx.random.key(3))
        assert mx.array_equal(a.x0, b.x0).item()

    def test_validation(self):
        logits = array_from_any(_random_logits((1, 3, 5), 7))
        with pytest.raises(ValueError, match="rank 3"):
            token_stats(logits[0])
        with pytest.raises(ValueError, match="temperature"):
            token_stats(logits, temperature=-1.0)
        with pytest.raises(ValueError, match="key"):
            token_stats(logits, temperature=1.0)
        with pytest.raises(ValueError, match="suppress_ids"):
            token_stats(logits, suppress_ids=[5])
        with pytest.raises(ValueError, match="suppress_ids"):
            token_stats(logits, suppress_ids=[-1])
        with pytest.raises(ValueError, match="suppress_ids"):
            token_stats(logits, suppress_ids=range(5))


class TestThresholdTransfer:
    @pytest.mark.parametrize("threshold", [0.0, 0.3, 0.7, 0.95, 1.0])
    def test_matches_fastdllm(self, threshold):
        conf, mask = _random_case(6, 32, 10)
        mask[0] = False  # an empty row
        out = threshold_transfer(array_from_any(conf), array_from_any(mask), threshold)
        ref = _fastdllm_transfer(conf, mask, None, threshold)
        assert np.array(out).tolist() == ref.tolist()

    def test_force_one_off_keeps_empty_selection(self):
        conf = mx.array([[0.1, 0.2, 0.3]])
        cand = mx.array([[True, True, True]])
        assert not mx.any(threshold_transfer(conf, cand, 0.9, force_one=False)).item()
        assert threshold_transfer(conf, cand, 0.9).tolist() == [[False, False, True]]

    def test_forces_exactly_one_on_ties(self):
        conf = mx.array([[0.4, 0.4, 0.4, 0.1]])
        cand = mx.array([[False, True, True, True]])
        # tie between positions 1 and 2: the first candidate wins.
        assert threshold_transfer(conf, cand, 0.9).tolist() == [[False, True, False, False]]

    def test_rows_are_independent(self):
        conf = mx.array([[0.9, 0.95, 0.1], [0.9, 0.95, 0.1]])
        cand = mx.array([[False, False, False], [True, True, True]])
        assert threshold_transfer(conf, cand, 0.5).tolist() == [
            [False, False, False],
            [True, True, False],
        ]

    def test_validation(self):
        conf, cand = mx.zeros((2, 3)), mx.ones((2, 3), dtype=mx.bool_)
        with pytest.raises(ValueError, match="threshold"):
            threshold_transfer(conf, cand, 1.5)
        with pytest.raises(ValueError, match="bool"):
            threshold_transfer(conf, cand.astype(mx.int32), 0.5)
        with pytest.raises(ValueError, match="shape"):
            threshold_transfer(conf, cand[:, :2], 0.5)
        with pytest.raises(ValueError, match="rank 2"):
            threshold_transfer(conf[0], cand[0], 0.5)


class TestTopkTransfer:
    def test_matches_fastdllm_per_row_k(self):
        conf, mask = _random_case(5, 24, 11)
        k = np.minimum(np.array([0, 1, 3, 7, 24]), mask.sum(axis=1))
        out = topk_transfer(array_from_any(conf), array_from_any(mask), array_from_any(k))
        ref = _fastdllm_transfer(conf, mask, k, None)
        assert np.array(out).tolist() == ref.tolist()

    def test_scalar_k(self):
        conf, mask = _random_case(3, 16, 12, p_mask=1.0)
        out = topk_transfer(array_from_any(conf), array_from_any(mask), 4)
        assert np.array(mx.sum(out, axis=1)).tolist() == [4, 4, 4]

    def test_k_larger_than_candidates(self):
        conf = mx.array([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]])
        cand = mx.array([[True, False, True], [False, False, False]])
        out = topk_transfer(conf, cand, mx.array([5, 2]))
        assert out.tolist() == [[True, False, True], [False, False, False]]

    def test_per_row_k_is_not_averaged(self):
        # Dream averages the quota over the batch; each row must get its own k.
        conf = mx.array([[0.1, 0.2, 0.3, 0.4]] * 2)
        cand = mx.ones((2, 4), dtype=mx.bool_)
        out = topk_transfer(conf, cand, mx.array([1, 3]))
        assert np.array(mx.sum(out, axis=1)).tolist() == [1, 3]

    def test_validation(self):
        conf, cand = mx.zeros((2, 3)), mx.ones((2, 3), dtype=mx.bool_)
        with pytest.raises(ValueError, match="k"):
            topk_transfer(conf, cand, -1)
        with pytest.raises(ValueError, match="k"):
            topk_transfer(conf, cand, mx.array([1, 2, 3]))
        with pytest.raises(ValueError, match="k"):
            topk_transfer(conf, cand, mx.array([1, -2]))
        with pytest.raises(ValueError, match="k"):
            topk_transfer(conf, cand, mx.array([1.0, 2.0]))


class TestFactorTransfer:
    @pytest.mark.parametrize("factor", [0.1, 0.5, 1.0, 2.0])
    def test_matches_fastdllm_without_off_by_one(self, factor):
        rng = np.random.default_rng(20)
        conf = (1 - rng.random((8, 16)) ** 3 * 0.5).astype(np.float32)  # mostly confident
        mask = rng.random((8, 16)) < 0.7
        mask[0] = False
        out = factor_transfer(array_from_any(conf), array_from_any(mask), factor)
        ref = _fastdllm_factor(conf, mask, factor, quirk=False)
        assert np.array(out).tolist() == ref.tolist()

    def test_last_failure_is_not_promoted(self):
        # sorted [0.99, 0.9, 0.1], factor 1: thresholds (-1, 0.667, 0.75) -> the
        # third fails. Fast-dLLM selects all three (off-by-one); we select two.
        conf = np.array([[0.1, 0.99, 0.9]], dtype=np.float32)
        mask = np.ones((1, 3), dtype=bool)
        assert _fastdllm_factor(conf, mask, 1.0, quirk=True).tolist() == [[True, True, True]]
        out = factor_transfer(array_from_any(conf), array_from_any(mask), 1.0)
        assert out.tolist() == [[False, True, True]]

    def test_always_commits_one(self):
        conf = mx.array([[0.01, 0.02, 0.03], [0.5, 0.5, 0.5]])
        cand = mx.array([[True, True, True], [False, False, False]])
        assert factor_transfer(conf, cand, 0.01).tolist() == [
            [False, False, True],
            [False, False, False],
        ]

    def test_validation(self):
        conf, cand = mx.zeros((1, 3)), mx.ones((1, 3), dtype=mx.bool_)
        with pytest.raises(ValueError, match="factor"):
            factor_transfer(conf, cand, 0.0)


class TestEntropyBoundTransfer:
    @pytest.mark.parametrize("bound", [0.0, 0.1, 1.0, 5.0])
    def test_matches_eb_sampler(self, bound):
        rng = np.random.default_rng(21)
        entropy = (rng.random((8, 20)) ** 2).astype(np.float32)
        mask = rng.random((8, 20)) < 0.7
        mask[0] = False
        out = entropy_bound_transfer(array_from_any(entropy), array_from_any(mask), bound)
        ref = _eb_accept(entropy, mask, bound)
        assert np.array(out).tolist() == ref.tolist()

    def test_zero_bound_commits_lowest_entropy_only(self):
        ent = mx.array([[0.5, 0.2, 0.9, 0.2]])
        cand = mx.ones((1, 4), dtype=mx.bool_)
        # tie at 0.2 between positions 1 and 3: position 1 first; the second 0.2
        # would add 0.2 > 0 to the budget.
        assert entropy_bound_transfer(ent, cand, 0.0).tolist() == [[False, True, False, False]]

    def test_rows_are_independent(self):
        ent = mx.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        cand = mx.array([[True, True, True], [False, True, False]])
        assert entropy_bound_transfer(ent, cand, 0.15).tolist() == [
            [True, True, False],
            [False, True, False],
        ]

    def test_validation(self):
        ent, cand = mx.zeros((1, 3)), mx.ones((1, 3), dtype=mx.bool_)
        with pytest.raises(ValueError, match="bound"):
            entropy_bound_transfer(ent, cand, -0.1)
        with pytest.raises(ValueError, match="bool"):
            entropy_bound_transfer(ent, cand.astype(mx.float32), 0.1)


class TestTransferSchedule:
    def test_matches_fastdllm(self):
        num = np.array([0, 1, 7, 32, 33])
        steps = 8
        base, rem = num // steps, num % steps
        ref = base[:, None] + (np.arange(steps)[None, :] < rem[:, None])
        out = transfer_schedule(array_from_any(num.astype(np.int32)), steps)
        assert out.dtype == mx.int32
        assert np.array(out).tolist() == ref.tolist()
        assert np.array(mx.sum(out, axis=1)).tolist() == num.tolist()

    def test_validation(self):
        with pytest.raises(ValueError, match="steps"):
            transfer_schedule(mx.array([4]), 0)
        with pytest.raises(ValueError, match="1D"):
            transfer_schedule(mx.array([[4]]), 2)
        with pytest.raises(ValueError, match="integer"):
            transfer_schedule(mx.array([4.0]), 2)
        with pytest.raises(ValueError, match="non-negative"):
            transfer_schedule(mx.array([-1]), 2)


class TestBlockRanges:
    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ((10, 8, 4), [(10, 14), (14, 18)]),
            ((10, 10, 4), [(10, 14), (14, 18), (18, 20)]),
            ((0, 5, 8), [(0, 5)]),
        ],
    )
    def test_prompt_relative(self, args, expected):
        assert block_ranges(*args) == expected

    def test_aligned_to_absolute_blocks(self):
        assert block_ranges(10, 8, 4, align=True) == [(10, 12), (12, 16), (16, 18)]

    def test_aligned_prompt_on_boundary(self):
        assert block_ranges(8, 8, 4, align=True) == [(8, 12), (12, 16)]

    def test_aligned_matches_block_causal_mask(self):
        # every aligned range stays inside one block of block_causal_mask.
        for start, end in block_ranges(5, 20, 4, align=True):
            assert (start // 4) == ((end - 1) // 4)

    def test_validation(self):
        with pytest.raises(ValueError, match="prompt_len"):
            block_ranges(-1, 4, 2)
        with pytest.raises(ValueError, match="gen_len"):
            block_ranges(0, 0, 2)
        with pytest.raises(ValueError, match="block_len"):
            block_ranges(0, 4, 0)


def _load_reference_loop() -> Any:
    note = Path(__file__).parents[1] / "docs" / "research" / "dllm-block-decoding.md"
    match = re.search(r"<!-- reference-loop -->\s*```python\n(.*?)```", note.read_text(), re.S)
    assert match, "reference loop block not found in the research note"
    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)
    return namespace["decode"]


class TestReferenceLoop:
    V, D, MASK = 40, 16, 39

    def _model(self, seed: int):
        rng = np.random.default_rng(seed)
        emb = array_from_any(rng.normal(size=(self.V, self.D)).astype(np.float32))
        proj = array_from_any(rng.normal(size=(self.D, self.V)).astype(np.float32))

        def model(x: mx.array) -> mx.array:
            h = emb[x]
            return 3.0 * (h + mx.mean(h, axis=1, keepdims=True)) @ proj

        return model

    @pytest.mark.parametrize(("threshold", "block_len"), [(0.9, 4), (0.0, 8), (1.0, 3)])
    def test_decodes_every_position(self, threshold, block_len):
        decode = _load_reference_loop()
        prompt = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
        out, forwards = decode(
            self._model(0),
            prompt,
            gen_len=12,
            block_len=block_len,
            mask_id=self.MASK,
            threshold=threshold,
        )
        assert out.shape == (2, 15)
        assert mx.array_equal(out[:, :3], prompt).item()
        assert not mx.any(out == self.MASK).item()
        n_blocks = len(block_ranges(3, 12, block_len))
        assert n_blocks <= forwards <= 12
        if threshold == 0.0:
            assert forwards == n_blocks  # everything commits in one step per block
        if threshold == 1.0:
            assert forwards == 12  # one forced commit per step
