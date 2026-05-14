"""Block streaming for low-RAM inference on Apple Silicon.

Stream transformer block weights from a memory-mapped safetensors file
into a single shared block module, so peak resident memory stays at
``~1 block`` instead of ``~num_blocks``.

The pattern fits any model whose state-dict has block-indexed keys of
the form ``f"{prefix}{i}.{rest}"`` (e.g.
``"transformer.transformer_blocks.0.attn.q_proj.weight"``). It works
because:

- Apple Silicon has unified memory, so there is no host-to-device copy.
- ``mx.load(path)`` memory-maps safetensors lazily — opening a 20 GB
  file costs ~40 MB RSS until individual arrays are touched.
- MLX has a single command queue, so per-block ``mx.clear_cache()`` +
  dropping references keeps the resident set bounded without explicit
  stream/event coordination.

Typical memory profile for a 22 B bf16 transformer with 48 blocks:
without streaming ~22 GB resident; with streaming ~1 block (~460 MB) +
non-block params + mmap metadata ≈ ~1 GB.

LoRA fusion is decoupled: pass a ``lora_fuser`` callable to
:class:`BlockStreamer` and it will be invoked per block. The fuser
receives the bound weights and a list of :class:`BlockLoraSource`
objects, and returns the fused weight list. This keeps quantization-
aware fusion strategies out of arsenal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn

__all__ = ["BlockLoraSource", "BlockStreamer", "LoraFuser"]


LoraFuser = Callable[
    [list[tuple[str, mx.array]], int, list["BlockLoraSource"]],
    list[tuple[str, mx.array]],
]
"""Callable that fuses LoRA deltas into a block's weight list.

Args:
    weights: ``[(param_name, array), ...]`` for the bound block.
    block_idx: Index of the block being bound.
    lora_sources: Sources to fuse (already filtered by ``has_block``).

Returns:
    The fused weight list (same shape; arrays may be replaced).
"""


class BlockLoraSource:
    """Per-block LoRA A/B matrices indexed by block index.

    Streams LoRA A/B pairs from a memory-mapped safetensors file and
    indexes them by block index so a fuser can apply the matching
    delta to a streamed block on the fly.

    Args:
        lora_path: Path to LoRA safetensors.
        block_prefix: Same prefix as the streamer (e.g.
            ``"transformer.transformer_blocks."``). LoRA keys after
            optional remapping must start with this prefix and be of
            the form ``f"{block_prefix}{idx}.{param}.lora_A.weight"``.
        strength: LoRA fusion strength (default 1.0).
        key_mapper: Optional callable that remaps raw safetensors keys
            to the model's state-dict naming (e.g. ComfyUI/diffusers →
            MLX). Returning ``None`` drops the key.
    """

    def __init__(
        self,
        lora_path: str | Path,
        block_prefix: str,
        strength: float = 1.0,
        key_mapper: Callable[[str], str | None] | None = None,
    ) -> None:
        self.strength = strength
        self.block_prefix = block_prefix
        self._lora_path = str(lora_path)
        loaded = mx.load(self._lora_path)
        assert isinstance(loaded, dict), f"expected dict from safetensors, got {type(loaded)}"
        self._lora_data: dict[str, mx.array] = cast(dict[str, mx.array], loaded)

        # block_idx -> param_name -> {"a": full_key, "b": full_key}
        self._block_keys: dict[int, dict[str, dict[str, str]]] = {}
        for raw_key in self._lora_data:
            model_key = key_mapper(raw_key) if key_mapper is not None else raw_key
            if model_key is None or not model_key.startswith(block_prefix):
                continue
            rest = model_key[len(block_prefix) :]
            idx_str, _, param_path = rest.partition(".")
            try:
                block_idx = int(idx_str)
            except ValueError:
                continue
            for suffix, slot in ((".lora_A.weight", "a"), (".lora_B.weight", "b")):
                if param_path.endswith(suffix):
                    param_name = param_path[: -len(suffix)]
                    self._block_keys.setdefault(block_idx, {}).setdefault(param_name, {})[slot] = (
                        raw_key
                    )
                    break

    def has_block(self, block_idx: int) -> bool:
        """True iff at least one matched A/B pair exists for ``block_idx``."""
        block = self._block_keys.get(block_idx)
        if not block:
            return False
        return any("a" in slots and "b" in slots for slots in block.values())

    def get_block_lora_dict(self, block_idx: int) -> dict[str, mx.array]:
        """Per-block LoRA dict shaped like ``{param.lora_A.weight: array, ...}``.

        Keys use the param name relative to the block (the same keys a
        :class:`BlockStreamer` would emit).
        """
        out: dict[str, mx.array] = {}
        block = self._block_keys.get(block_idx, {})
        for param_name, slots in block.items():
            if "a" not in slots or "b" not in slots:
                continue
            out[f"{param_name}.lora_A.weight"] = self._lora_data[slots["a"]]
            out[f"{param_name}.lora_B.weight"] = self._lora_data[slots["b"]]
        return out

    def close(self) -> None:
        self._lora_data = {}
        self._block_keys = {}


class BlockStreamer:
    """Stream transformer block weights from mmap'd safetensors.

    Args:
        weight_paths: One or more ``.safetensors`` paths whose union
            contains every key for the streamed blocks. Loaded via
            :func:`mlx.core.load` which memory-maps the file.
        block_prefix: State-dict key prefix that identifies block
            weights, e.g. ``"transformer.transformer_blocks."``. Keys
            of the form ``f"{block_prefix}{i}.{rest}"`` (where ``i`` is
            an integer) are treated as block ``i``'s weights.
        lora_fuser: Optional callable to fuse LoRA deltas into the
            bound weights. See :data:`LoraFuser`. If ``None``, the
            ``lora_sources`` argument to :meth:`bind` is ignored.

    Notes:
        After construction, :attr:`block_count` returns the number of
        distinct block indices found, and :meth:`block_keys(i)` lists
        the per-block parameter names that will be bound by
        :meth:`bind`. The mmap'd dict is held until :meth:`close` is
        called.
    """

    def __init__(
        self,
        weight_paths: str | Path | Iterable[str | Path],
        block_prefix: str,
        lora_fuser: LoraFuser | None = None,
    ) -> None:
        if isinstance(weight_paths, str | Path):
            weight_paths = [weight_paths]
        self._weight_paths = [str(p) for p in weight_paths]
        self._block_prefix = block_prefix
        self._lora_fuser = lora_fuser

        # Merge all safetensors files. mx.load mmaps each one, so cost
        # is roughly metadata-only until individual arrays are touched.
        self._weights = self._reload_dict()

        # Build per-block key map: idx -> list[(full_key, param_name)].
        self._block_key_map: dict[int, list[tuple[str, str]]] = {}
        for full_key in self._weights:
            if not full_key.startswith(self._block_prefix):
                continue
            rest = full_key[len(self._block_prefix) :]
            idx_str, _, param_name = rest.partition(".")
            try:
                block_idx = int(idx_str)
            except ValueError:
                continue
            self._block_key_map.setdefault(block_idx, []).append((full_key, param_name))

        if not self._block_key_map:
            raise ValueError(
                f"BlockStreamer found no keys matching prefix {block_prefix!r} "
                f"in {self._weight_paths!r}. Check the prefix or the safetensors content."
            )

    @property
    def block_count(self) -> int:
        """Number of distinct block indices discovered in the safetensors."""
        return len(self._block_key_map)

    @property
    def block_prefix(self) -> str:
        return self._block_prefix

    def block_keys(self, idx: int) -> list[str]:
        """Per-block parameter names (without the ``{prefix}{idx}.`` part)."""
        if idx not in self._block_key_map:
            raise KeyError(f"block {idx} not in streamer (have {sorted(self._block_key_map)})")
        return [param_name for _full, param_name in self._block_key_map[idx]]

    def bind(
        self,
        block: nn.Module,
        idx: int,
        evict_previous: int | None = None,
        lora_sources: list[BlockLoraSource] | None = None,
    ) -> None:
        """Load block ``idx``'s weights into ``block`` in-place.

        After this returns, ``block``'s parameters reference the
        safetensors-mapped arrays for index ``idx``. A subsequent
        :meth:`bind` to a different ``idx`` rebinds them, releasing the
        previous arrays from the block's parameter tree.

        Args:
            block: Target module. Its parameter tree must match the
                keys returned by :meth:`block_keys`.
            idx: Block index to load.
            evict_previous: If given, drop the cached array references
                for that block index from the internal weight dict
                before binding. The streamer holds refs to every
                block's weights; without eviction, those refs prevent
                GC even after the bound block is replaced. For
                streaming inference, pass the previously-bound index
                so peak resident memory stays at ~one block.
            lora_sources: Optional LoRA sources to fuse into the bound
                weights. Requires a ``lora_fuser`` at construction.
        """
        if idx not in self._block_key_map:
            raise KeyError(f"block {idx} not in streamer")
        if evict_previous is not None and evict_previous in self._block_key_map:
            for full_key, _ in self._block_key_map[evict_previous]:
                self._weights.pop(full_key, None)
        # Re-mmap if the requested block's keys have already been evicted
        # (typical after a full forward sweep through every block).
        sample_key = self._block_key_map[idx][0][0]
        if sample_key not in self._weights:
            self._weights = self._reload_dict()
        weights = [
            (param_name, self._weights[full_key])
            for full_key, param_name in self._block_key_map[idx]
        ]

        if lora_sources and self._lora_fuser is not None:
            active = [src for src in lora_sources if src.has_block(idx)]
            if active:
                weights = self._lora_fuser(weights, idx, active)

        block.load_weights(weights, strict=True)

    def _reload_dict(self) -> dict[str, mx.array]:
        """Re-mmap all weight files into a fresh dict."""
        merged: dict[str, mx.array] = {}
        for path in self._weight_paths:
            loaded = mx.load(path)
            assert isinstance(loaded, dict), f"expected dict from safetensors, got {type(loaded)}"
            merged.update(cast(dict[str, mx.array], loaded))
        return merged

    def close(self) -> None:
        """Release the mmap'd dict. After this the streamer is unusable."""
        self._weights = {}
        self._block_key_map = {}
