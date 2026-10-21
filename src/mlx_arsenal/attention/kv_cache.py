"""Sliding-window KV cache with an attention sink, for chunked causal generation.

Autoregressive video DiTs (Self-Forcing, CausVid, Matrix-Game, SkyReels-V2
diffusion forcing) and streaming LLMs generate a sequence chunk by chunk.
Each layer keeps the keys/values of earlier chunks, attends to the first
``sink`` tokens plus the most recent ``window - sink`` tokens, and evicts
the rest. Diffusion-style models also rewrite the current chunk several
times (one write per denoising pass, then a final commit) at the same
positions, which a plain append-only cache cannot express.

The cache works in absolute token positions. RoPE, the attention call and
the frame-to-token arithmetic stay with the caller.
"""

import mlx.core as mx


class RollingKVCache:
    """Per-layer KV cache with sink tokens, a sliding window and in-place rewrites.

    ``update(k, v, start=p)`` writes ``L`` tokens at absolute positions
    ``[p, p + L)`` and **truncates the history at** ``p + L``: writing the
    same chunk again (the next denoising pass) replaces it rather than
    appending. It returns the keys/values to attend to: the first ``sink``
    tokens of the history followed by its last ``window - sink`` tokens (the
    whole history while it is at most ``window`` tokens long, or always when
    ``window`` is ``None``). Tokens between the two are evicted for good.

    Typical use, one cache per layer::

        cache = RollingKVCache(window=W * frame_len, sink=S * frame_len)
        k_att, v_att = cache.update(k_rot, v, start=chunk_index * frame_len)
        out = mx.fast.scaled_dot_product_attention(q_rot, k_att, v_att, scale=scale)

    Fixed-sink RoPE (re-rotating the sink keys to sit just before the window,
    as in Matrix-Game 4) is model-specific: pass the pre-RoPE keys as
    ``k_raw`` and re-rotate :attr:`sink_raw_keys` yourself, then replace the
    first ``sink`` keys of the returned tensor.

    Args:
        window: Maximum number of tokens attended to, sink included.
            ``None`` keeps the whole history.
        sink: Number of leading tokens that are never evicted.
        seq_axis: Sequence axis of ``k`` / ``v``. ``2`` matches the
            ``(B, H, L, D)`` layout of ``mx.fast.scaled_dot_product_attention``;
            use ``1`` for ``(B, L, H, D)``.
    """

    def __init__(self, window: int | None = None, sink: int = 0, seq_axis: int = 2):
        if sink < 0:
            raise ValueError(f"sink must be >= 0, got {sink}")
        if window is not None and window <= sink:
            raise ValueError(f"window ({window}) must be larger than sink ({sink})")
        self.window = window
        self.sink = sink
        self.seq_axis = seq_axis
        self.reset()

    def reset(self) -> None:
        """Forget all tokens (start of a new sequence)."""
        self._k: mx.array | None = None
        self._v: mx.array | None = None
        self._sink_raw: mx.array | None = None
        self._end = 0
        self._evicted = 0

    @property
    def end(self) -> int:
        """Absolute position one past the last written token."""
        return self._end

    @property
    def num_evicted(self) -> int:
        """Number of tokens evicted between the sink and the window."""
        return self._evicted

    @property
    def sink_raw_keys(self) -> mx.array | None:
        """Pre-RoPE keys of the sink tokens, or ``None`` if not all were given.

        Available once every write that touched a sink position passed
        ``k_raw``.
        """
        if self._sink_raw is None or self._sink_raw.shape[self.seq_axis] < min(
            self.sink, self._end
        ):
            return None
        return self._sink_raw

    def positions(self) -> mx.array:
        """Absolute positions of the tokens returned by the last :meth:`update`."""
        if not self._evicted:
            return mx.arange(self._end, dtype=mx.int32)
        tail = mx.arange(self.sink + self._evicted, self._end, dtype=mx.int32)
        return mx.concatenate([mx.arange(self.sink, dtype=mx.int32), tail])

    def _take(self, x: mx.array, start: int, stop: int | None = None) -> mx.array:
        idx = [slice(None)] * x.ndim
        idx[self.seq_axis] = slice(start, stop)
        return x[tuple(idx)]

    def update(
        self, k: mx.array, v: mx.array, *, start: int, k_raw: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        """Write a chunk at absolute position ``start`` and return the attended K/V.

        Args:
            k: Keys of the chunk (as attended, e.g. after RoPE).
            v: Values of the chunk, same shape as ``k``.
            start: Absolute position of the first token of the chunk. It may
                rewrite resident tokens (``start < end``) but not leave a gap
                (``start > end``) or rewrite evicted ones.
            k_raw: Optional pre-RoPE keys of the chunk, kept for the sink
                positions only (see :attr:`sink_raw_keys`).

        Returns:
            ``(k_att, v_att)``: sink tokens followed by the recent window,
            in position order. :meth:`positions` gives their positions.

        Raises:
            ValueError: On a gap, a rewrite of evicted tokens, or a chunk
                longer than ``window - sink`` (its own tokens would be evicted).
        """
        n = k.shape[self.seq_axis]
        if start > self._end:
            raise ValueError(f"start={start} leaves a gap after end={self._end}")
        if self._evicted and self.sink < start < self.sink + self._evicted:
            raise ValueError(
                f"start={start} rewrites evicted tokens [{self.sink}, {self.sink + self._evicted})"
            )
        if self.window is not None and n > self.window - self.sink:
            raise ValueError(
                f"chunk of {n} tokens exceeds the recent window ({self.window - self.sink} tokens)"
            )

        if start < self.sink:
            # Raw sink keys stay valid only if every write to a sink position supplied them.
            prev = self._sink_raw
            covered = start == 0 or (prev is not None and prev.shape[self.seq_axis] >= start)
            if k_raw is None or not covered:
                self._sink_raw = None
            else:
                parts = [self._take(k_raw, 0, self.sink - start)]
                if start > 0 and prev is not None:
                    parts.insert(0, self._take(prev, 0, start))
                self._sink_raw = mx.concatenate(parts, axis=self.seq_axis)

        # Buffer index of `start`. Writing at or inside the sink truncates every
        # later token, so nothing is evicted any more; past the sink, positions
        # shift by the evicted count.
        if start <= self.sink:
            cut = start
            self._evicted = 0
        else:
            cut = start - self._evicted
        if self._k is None or self._v is None:
            self._k, self._v = k, v
        else:
            self._k = mx.concatenate([self._take(self._k, 0, cut), k], axis=self.seq_axis)
            self._v = mx.concatenate([self._take(self._v, 0, cut), v], axis=self.seq_axis)
        self._end = start + n

        size = self._k.shape[self.seq_axis]
        if self.window is not None and size > self.window:
            drop = size - self.window
            keep_from = self.sink + drop
            self._k = mx.concatenate(
                [self._take(self._k, 0, self.sink), self._take(self._k, keep_from)],
                axis=self.seq_axis,
            )
            self._v = mx.concatenate(
                [self._take(self._v, 0, self.sink), self._take(self._v, keep_from)],
                axis=self.seq_axis,
            )
            self._evicted += drop
        return self._k, self._v
