"""Tests for RollingKVCache (sliding-window KV cache with attention sink)."""

import mlx.core as mx
import pytest

from mlx_arsenal.attention import RollingKVCache


def _tokens(start: int, n: int, *, tag: float = 0.0, seq_axis: int = 2) -> mx.array:
    """(B=1, H=2, L, D=3) tensor whose values encode absolute positions (+ a tag)."""
    pos = mx.arange(start, start + n, dtype=mx.float32) + tag
    x = mx.broadcast_to(pos[None, None, :, None], (1, 2, n, 3))
    return mx.moveaxis(x, 2, seq_axis)


class _Oracle:
    """Full history, truncated at each write; attends sink head + recent tail."""

    def __init__(self, window, sink):
        self.window, self.sink, self.hist = window, sink, None

    def update(self, k, start):
        self.hist = k if self.hist is None else mx.concatenate([self.hist[:, :, :start], k], axis=2)
        n = self.hist.shape[2]
        if self.window is None or n <= self.window:
            return self.hist
        tail = self.hist[:, :, n - (self.window - self.sink) :]
        return mx.concatenate([self.hist[:, :, : self.sink], tail], axis=2)


def _rollout(window, sink, chunk, n_chunks, passes):
    """Denoise-then-commit schedule: `passes` rewrites of each chunk at the same start."""
    for c in range(n_chunks):
        for p in range(passes):
            yield c * chunk, chunk, float(p) / 10


class TestRollingKVCache:
    @pytest.mark.parametrize(
        ("window", "sink", "chunk", "passes"),
        [(None, 0, 4, 1), (12, 0, 4, 1), (12, 4, 4, 3), (12, 2, 3, 2), (9, 3, 2, 4)],
    )
    def test_matches_oracle(self, window, sink, chunk, passes):
        cache = RollingKVCache(window=window, sink=sink)
        oracle = _Oracle(window, sink)
        for start, n, tag in _rollout(window, sink, chunk, 12, passes):
            k = _tokens(start, n, tag=tag)
            k_att, v_att = cache.update(k, -k, start=start)
            expected = oracle.update(k, start)
            assert mx.array_equal(k_att, expected)
            assert mx.array_equal(v_att, -expected)
            assert cache.end == start + n

    def test_memory_stays_flat(self):
        cache = RollingKVCache(window=10, sink=2)
        for c in range(200):
            k_att, _ = cache.update(_tokens(4 * c, 4), _tokens(4 * c, 4), start=4 * c)
        assert k_att.shape[2] == 10
        assert cache.num_evicted == 800 - 10

    def test_positions(self):
        cache = RollingKVCache(window=8, sink=2)
        for c in range(4):
            k_att, _ = cache.update(_tokens(3 * c, 3), _tokens(3 * c, 3), start=3 * c)
        pos = cache.positions()
        assert pos.tolist() == [0, 1, 6, 7, 8, 9, 10, 11]
        assert mx.array_equal(k_att[0, 0, :, 0], pos.astype(mx.float32))

    def test_seq_axis_one(self):
        a = RollingKVCache(window=6, sink=2, seq_axis=1)
        b = RollingKVCache(window=6, sink=2, seq_axis=2)
        for c in range(5):
            ka, _ = a.update(
                _tokens(2 * c, 2, seq_axis=1), _tokens(2 * c, 2, seq_axis=1), start=2 * c
            )
            kb, _ = b.update(_tokens(2 * c, 2), _tokens(2 * c, 2), start=2 * c)
            assert mx.array_equal(mx.moveaxis(ka, 1, 2), kb)

    def test_rewrite_is_idempotent(self):
        cache = RollingKVCache(window=8, sink=2)
        for c in range(5):
            cache.update(_tokens(2 * c, 2), _tokens(2 * c, 2), start=2 * c)
        k1, _ = cache.update(_tokens(8, 2), _tokens(8, 2), start=8)
        k2, _ = cache.update(_tokens(8, 2), _tokens(8, 2), start=8)
        assert mx.array_equal(k1, k2)
        assert cache.end == 10

    def test_rewrite_truncates_later_tokens(self):
        cache = RollingKVCache()
        cache.update(_tokens(0, 6), _tokens(0, 6), start=0)
        k_att, _ = cache.update(_tokens(2, 2, tag=0.5), _tokens(2, 2), start=2)
        assert cache.end == 4
        assert k_att[0, 0, :, 0].tolist() == [0.0, 1.0, 2.5, 3.5]

    def test_rewrite_inside_sink_after_eviction(self):
        cache = RollingKVCache(window=6, sink=2)
        for c in range(5):
            cache.update(_tokens(2 * c, 2), _tokens(2 * c, 2), start=2 * c)
        k_att, _ = cache.update(_tokens(1, 2, tag=0.5), _tokens(1, 2), start=1)
        assert cache.end == 3 and cache.num_evicted == 0
        assert k_att[0, 0, :, 0].tolist() == [0.0, 1.5, 2.5]

    def test_rejects_evicted_rewrite(self):
        cache = RollingKVCache(window=6, sink=2)
        for c in range(5):
            cache.update(_tokens(2 * c, 2), _tokens(2 * c, 2), start=2 * c)
        with pytest.raises(ValueError, match="evicted"):
            cache.update(_tokens(4, 2), _tokens(4, 2), start=4)

    def test_rejects_gap(self):
        cache = RollingKVCache()
        cache.update(_tokens(0, 2), _tokens(0, 2), start=0)
        with pytest.raises(ValueError, match="gap"):
            cache.update(_tokens(3, 2), _tokens(3, 2), start=3)

    @pytest.mark.parametrize(("start", "n"), [(0, 7), (2, 5), (1, 6)])
    def test_rejects_chunk_whose_tokens_would_be_evicted(self, start, n):
        cache = RollingKVCache(window=6, sink=2)
        if start:
            cache.update(_tokens(0, start), _tokens(0, start), start=0)
        with pytest.raises(ValueError, match="window"):
            cache.update(_tokens(start, n), _tokens(start, n), start=start)

    def test_prefill_through_the_sink_fills_the_window(self):
        cache = RollingKVCache(window=6, sink=2)
        k_att, _ = cache.update(_tokens(0, 6), _tokens(0, 6), start=0)
        assert k_att[0, 0, :, 0].tolist() == [float(p) for p in range(6)]
        assert cache.num_evicted == 0
        k_att, _ = cache.update(_tokens(6, 4), _tokens(6, 4), start=6)
        assert cache.positions().tolist() == [0, 1, 6, 7, 8, 9]
        assert k_att[0, 0, :, 0].tolist() == [0.0, 1.0, 6.0, 7.0, 8.0, 9.0]

    def test_rejects_negative_start(self):
        cache = RollingKVCache()
        cache.update(_tokens(0, 4), _tokens(0, 4), start=0)
        with pytest.raises(ValueError, match="start"):
            cache.update(_tokens(0, 1), _tokens(0, 1), start=-1)

    def test_rejects_length_mismatch(self):
        cache = RollingKVCache()
        with pytest.raises(ValueError, match="same length"):
            cache.update(_tokens(0, 2), _tokens(0, 3), start=0)
        with pytest.raises(ValueError, match="same length"):
            cache.update(_tokens(0, 2), _tokens(0, 2), start=0, k_raw=_tokens(0, 1))

    def test_failed_update_leaves_cache_unchanged(self):
        cache = RollingKVCache(window=6, sink=2)
        for c in range(5):
            k = _tokens(2 * c, 2)
            cache.update(k, k, start=2 * c, k_raw=k)
        before = (cache.end, cache.num_evicted, cache.positions().tolist())
        bad = mx.zeros((1, 3, 2, 3))  # other head count: the concatenate fails
        with pytest.raises(ValueError):
            cache.update(bad, bad, start=0, k_raw=bad)
        assert (cache.end, cache.num_evicted, cache.positions().tolist()) == before
        assert cache.sink_raw_keys is not None
        k_att, _ = cache.update(_tokens(10, 2), _tokens(10, 2), start=10)
        assert k_att[0, 0, :, 0].tolist() == [0.0, 1.0, 8.0, 9.0, 10.0, 11.0]

    def test_shorter_rewrite_does_not_restore_evicted_tokens(self):
        cache = RollingKVCache(window=6, sink=2)
        for c in range(5):
            cache.update(_tokens(2 * c, 2), _tokens(2 * c, 2), start=2 * c)
        k_att, _ = cache.update(_tokens(8, 1), _tokens(8, 1), start=8)
        assert cache.positions().tolist() == [0, 1, 6, 7, 8]
        assert k_att[0, 0, :, 0].tolist() == [0.0, 1.0, 6.0, 7.0, 8.0]

    @pytest.mark.parametrize(("window", "sink"), [(4, 4), (4, 6), (None, -1)])
    def test_rejects_bad_config(self, window, sink):
        with pytest.raises(ValueError):
            RollingKVCache(window=window, sink=sink)

    def test_sink_raw_keys(self):
        cache = RollingKVCache(window=6, sink=2)
        assert cache.sink_raw_keys is None
        for c in range(5):
            k = _tokens(2 * c, 2)
            cache.update(k, k, start=2 * c, k_raw=k + 100)
        raw = cache.sink_raw_keys
        assert raw is not None
        assert raw[0, 0, :, 0].tolist() == [100.0, 101.0]

    def test_sink_raw_keys_partial_sink_writes(self):
        cache = RollingKVCache(window=8, sink=3)
        for start in range(0, 6, 2):
            k = _tokens(start, 2)
            cache.update(k, k, start=start, k_raw=k + 100)
        raw = cache.sink_raw_keys
        assert raw is not None
        assert raw[0, 0, :, 0].tolist() == [100.0, 101.0, 102.0]

    def test_sink_raw_keys_invalidated_without_k_raw(self):
        cache = RollingKVCache(window=6, sink=2)
        k = _tokens(0, 2)
        cache.update(k, k, start=0, k_raw=k)
        cache.update(k, k, start=0)
        assert cache.sink_raw_keys is None

    def test_reset(self):
        cache = RollingKVCache(window=6, sink=2)
        for c in range(5):
            cache.update(_tokens(2 * c, 2), _tokens(2 * c, 2), start=2 * c)
        cache.reset()
        assert cache.end == 0 and cache.num_evicted == 0
        k_att, _ = cache.update(_tokens(0, 2), _tokens(0, 2), start=0)
        assert k_att.shape[2] == 2
