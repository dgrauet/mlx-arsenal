"""Safetensors state-dict loader with optional :class:`SDOps` key remapping."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import mlx.core as mx

from mlx_arsenal.loader.sd_ops import KeyValueOperationResult, SDOps
from mlx_arsenal.loader.state_dict import StateDict

__all__ = ["SafetensorsStateDictLoader", "read_safetensors_metadata"]


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    """Read the ``__metadata__`` block from a safetensors file.

    Safetensors layout: 8 little-endian bytes hold the JSON header
    length, then the header (UTF-8 JSON), then the raw tensor bytes.
    The optional ``__metadata__`` key in the header carries arbitrary
    string-keyed string-valued model metadata (HF convention).

    Args:
        path: Path to a ``.safetensors`` file.

    Returns:
        The metadata dict, or an empty dict if the file has no
        ``__metadata__`` block.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the header is malformed.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length: {header_len}")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"truncated safetensors header in {path}")
        header = json.loads(header_bytes.decode("utf-8"))
    meta = header.get("__metadata__", {})
    return meta if isinstance(meta, dict) else {}


class SafetensorsStateDictLoader:
    """Load state-dicts from one or more safetensors shards.

    Uses :func:`mlx.core.load` for the actual tensor loading (which
    memory-maps the file). The :class:`SDOps` chain runs over each
    raw key — keys it drops are silently skipped, keys it rewrites
    end up in the output dict under their new name, and the optional
    key/value transforms run after the rename step.

    The loader is stateless; create one and reuse it.
    """

    def load(
        self,
        path: str | Path | list[str | Path],
        sd_ops: SDOps | None = None,
    ) -> StateDict:
        """Load one or more safetensors shards into a :class:`StateDict`.

        Args:
            path: Single path or list of paths to merge. Later shards
                with overlapping keys overwrite earlier ones (matches
                ``dict.update`` semantics).
            sd_ops: Optional rename/match/transform chain.

        Returns:
            The merged :class:`StateDict`. ``size`` accumulates each
            tensor's ``nbytes``; ``dtype`` collects every distinct
            dtype seen.
        """
        sd: dict[str, mx.array] = {}
        size = 0
        dtypes: set[mx.Dtype] = set()

        shards = path if isinstance(path, list) else [path]
        for shard in shards:
            loaded = mx.load(str(shard))
            assert isinstance(loaded, dict), (
                f"expected dict from safetensors, got {type(loaded)} for {shard!r}"
            )
            weights = cast(dict[str, mx.array], loaded)

            for raw_key, raw_value in weights.items():
                if sd_ops is not None:
                    expected_key = sd_ops.apply_to_key(raw_key)
                    if expected_key is None:
                        continue
                    pairs = sd_ops.apply_to_key_value(expected_key, raw_value)
                else:
                    pairs = [KeyValueOperationResult(raw_key, raw_value)]

                for key, val in pairs:
                    size += val.nbytes
                    dtypes.add(val.dtype)
                    sd[key] = val

        return StateDict(sd=sd, size=size, dtype=dtypes)
