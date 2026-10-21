"""Tests for apply_block_edits (LLaDA2.2 DELETE / SPLIT block edits).

The oracle transcribes ``JointThresholdInDel._apply_edit_operations`` from
SGLang (Apache-2.0, python/sglang/srt/dllm/algorithm/joint_threshold_indel.py,
PR #31773).
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal.diffusion import apply_block_edits

MASK, DEL, SPLIT = 9, 7, 8


def _ref_apply(new, old, is_orig, prompt_len, block_size):
    tokens = list(old[:prompt_len])
    orig = list(is_orig[:prompt_len])
    for i in range(prompt_len, block_size):
        t = new[i]
        if t == DEL:
            continue
        if t == SPLIT:
            tokens.extend([MASK, old[i]])
            orig.extend([False, is_orig[i]])
        else:
            tokens.append(t)
            orig.append(is_orig[i])
    if len(tokens) > block_size:
        tokens, orig = tokens[:block_size], orig[:block_size]
    else:
        pad = block_size - len(tokens)
        tokens += [MASK] * pad
        orig += [False] * pad
    return tokens, orig


def _apply(old, new, orig, prompt_len):
    return apply_block_edits(
        mx.array(old, dtype=mx.int32),
        mx.array(new, dtype=mx.int32),
        mx.array(orig),
        prompt_len=prompt_len,
        mask_id=MASK,
        delete_id=DEL,
        split_id=SPLIT,
    )


class TestApplyBlockEdits:
    def test_delete_shifts_left_and_pads_with_new_mask(self):
        old = [[1, 2, 3, MASK, MASK]]
        new = [[1, DEL, 3, 4, MASK]]
        orig = [[False, False, False, True, True]]
        tokens, is_orig = _apply(old, new, orig, 0)
        assert tokens.tolist() == [[1, 3, 4, MASK, MASK]]
        assert is_orig.tolist() == [[False, False, True, True, False]]

    def test_split_inserts_mask_before_old_token(self):
        old = [[1, 2, 3, MASK]]
        new = [[1, SPLIT, 3, MASK]]
        orig = [[False, False, False, True]]
        tokens, is_orig = _apply(old, new, orig, 0)
        assert tokens.tolist() == [[1, MASK, 2, 3]]  # the trailing mask falls off
        assert is_orig.tolist() == [[False, False, False, False]]

    def test_prompt_is_copied_from_old(self):
        old = [[5, 6, 1, 2]]
        new = [[DEL, SPLIT, DEL, 2]]
        orig = [[False, False, False, False]]
        tokens, _ = _apply(old, new, orig, 2)
        assert tokens.tolist() == [[5, 6, 2, MASK]]

    def test_no_edits_is_identity_on_new(self):
        old = [[1, MASK, MASK, 4]]
        new = [[1, 2, MASK, 4]]
        orig = [[False, True, True, False]]
        tokens, is_orig = _apply(old, new, orig, 1)
        assert tokens.tolist() == new
        assert is_orig.tolist() == orig

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_reference_batched(self, seed):
        rng = np.random.default_rng(seed)
        B, L = 6, 12
        vocab = [1, 2, 3, 4, 5, MASK, DEL, SPLIT]
        old = rng.choice([1, 2, 3, 4, 5, MASK], size=(B, L))
        new = np.where(rng.random((B, L)) < 0.5, rng.choice(vocab, size=(B, L)), old)
        orig = (old == MASK) & (rng.random((B, L)) < 0.8)
        prompt_len = rng.integers(0, L // 2, size=B)
        tokens, is_orig = apply_block_edits(
            mx.array(old, dtype=mx.int32),
            mx.array(new, dtype=mx.int32),
            mx.array(orig),
            prompt_len=mx.array(prompt_len, dtype=mx.int32),
            mask_id=MASK,
            delete_id=DEL,
            split_id=SPLIT,
        )
        for b in range(B):
            ref_t, ref_o = _ref_apply(
                new[b].tolist(), old[b].tolist(), orig[b].tolist(), int(prompt_len[b]), L
            )
            assert tokens[b].tolist() == ref_t, b
            assert is_orig[b].tolist() == ref_o, b

    def test_preserves_dtype_and_shape(self):
        old = mx.array([[1, 2, 3]], dtype=mx.int64)
        tokens, is_orig = apply_block_edits(
            old, old, mx.zeros((1, 3), dtype=mx.bool_), prompt_len=0,
            mask_id=MASK, delete_id=DEL, split_id=SPLIT,
        )  # fmt: skip
        assert tokens.dtype == mx.int64 and tokens.shape == (1, 3)
        assert is_orig.dtype == mx.bool_

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            _apply([[1, 2]], [[1, 2, 3]], [[False, False]], 0)

    def test_rejects_non_bool_is_orig(self):
        with pytest.raises(ValueError, match="bool"):
            apply_block_edits(
                mx.array([[1, 2]]), mx.array([[1, 2]]), mx.array([[0, 1]]), prompt_len=0,
                mask_id=MASK, delete_id=DEL, split_id=SPLIT,
            )  # fmt: skip


def _load_indel_step():
    import re
    from pathlib import Path
    from typing import Any

    note = Path(__file__).parents[1] / "docs" / "research" / "dllm-block-decoding.md"
    match = re.search(r"<!-- indel-step -->\s*```python\n(.*?)```", note.read_text(), re.S)
    assert match, "InDel step block not found in the research note"
    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)
    return namespace["indel_step"]


def _ref_step(logits, block, is_orig, prompt_len, threshold, edit_threshold, final):
    """SGLang JointThresholdInDel.step for one row, minus anti-loop and budget."""
    L = len(block)
    x = logits.argmax(-1)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    p = probs[np.arange(L), x]
    prompt = np.arange(L) < prompt_len
    mask_index = ~prompt & (block == MASK)
    orig = is_orig & mask_index
    n_orig, n_total = int(orig.sum()), int(mask_index.sum())
    if not final:
        for i in np.nonzero(orig & (x == MASK))[0]:
            lg = logits[i].copy()
            lg[MASK] = -np.inf
            x[i] = lg.argmax()
            p[i] = probs[i, x[i]]
    m2t = np.zeros(L, bool)
    if n_total > 0:
        conf = np.where(mask_index, p, -np.inf)
        if final:
            m2t = mask_index & ~prompt
        elif n_orig > 0:
            need = 1 + n_total - n_orig
            high = (conf > threshold) & mask_index
            if high.sum() >= need:
                m2t = high
            else:
                m2t[np.argsort(-conf, kind="stable")[: min(need, n_total)]] = True
        else:
            m2t = mask_index
    t2t = (p > edit_threshold) & (block != x) & ~mask_index & ~prompt
    write = m2t | t2t
    if final:
        for i in np.nonzero(write & np.isin(x, [MASK, DEL, SPLIT]))[0]:
            lg = logits[i].copy()
            lg[[MASK, DEL, SPLIT]] = -np.inf
            x[i] = lg.argmax()
    new = np.where(write, x, block)
    return _ref_apply(new.tolist(), block.tolist(), is_orig.tolist(), prompt_len, L)


class TestInDelStepRecipe:
    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("final", [False, True])
    def test_matches_sglang_step(self, seed, final):
        step = _load_indel_step()
        rng = np.random.default_rng(seed)
        B, L, V = 3, 10, 12
        block = rng.choice([1, 2, 3, 4, 5, MASK], size=(B, L), p=[0.1] * 5 + [0.5])
        is_orig = (block == MASK) & (rng.random((B, L)) < 0.7)
        prompt_len = int(rng.integers(0, 3))
        logits = rng.normal(size=(B, L, V)) * 2.0
        logits[..., [MASK, DEL, SPLIT]] += rng.normal(1.0, 1.0, size=(B, L, 3))
        tokens, orig = step(
            mx.array(logits, dtype=mx.float32), mx.array(block, dtype=mx.int32),
            mx.array(is_orig), prompt_len=prompt_len, mask_id=MASK, delete_id=DEL,
            split_id=SPLIT, threshold=0.3, editing_threshold=0.2, final=final,
        )  # fmt: skip
        for b in range(B):
            ref_t, ref_o = _ref_step(
                logits[b].astype(np.float32).astype(np.float64), block[b], is_orig[b],
                prompt_len, 0.3, 0.2, final,
            )  # fmt: skip
            assert tokens[b].tolist() == ref_t, (b, final)
            assert orig[b].tolist() == ref_o, (b, final)
