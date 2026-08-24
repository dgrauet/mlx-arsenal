"""Tests for block streaming module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_arsenal.streaming import BlockLoraSource, BlockStreamer


def _block_state_dict(prefix: str, num_blocks: int, dim: int = 4) -> dict[str, mx.array]:
    """Synthetic state-dict with ``num_blocks`` linear-shaped blocks."""
    sd: dict[str, mx.array] = {}
    for i in range(num_blocks):
        sd[f"{prefix}{i}.linear.weight"] = mx.ones((dim, dim)) * (i + 1)
        sd[f"{prefix}{i}.linear.bias"] = mx.zeros((dim,)) + i
    # Non-block weights that the streamer should ignore.
    sd["embed.weight"] = mx.zeros((dim, dim))
    return sd


def _save_safetensors(tmp_path, name: str, sd: dict[str, mx.array]) -> str:
    path = str(tmp_path / name)
    mx.save_safetensors(path, sd)
    # mx.save_safetensors appends .safetensors if missing
    if not path.endswith(".safetensors"):
        path = path + ".safetensors"
    return path


class _Block(nn.Module):
    """Minimal block whose parameter tree matches ``{prefix}{i}.linear.{weight,bias}``."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)


class TestBlockStreamerDiscovery:
    def test_block_count_and_keys(self, tmp_path):
        sd = _block_state_dict("blocks.", num_blocks=3)
        path = _save_safetensors(tmp_path, "w", sd)

        streamer = BlockStreamer(path, block_prefix="blocks.")
        assert streamer.block_count == 3
        assert streamer.block_prefix == "blocks."
        assert sorted(streamer.block_keys(0)) == ["linear.bias", "linear.weight"]

    def test_unknown_prefix_raises(self, tmp_path):
        sd = _block_state_dict("blocks.", num_blocks=2)
        path = _save_safetensors(tmp_path, "w", sd)
        with pytest.raises(ValueError, match="no keys matching prefix"):
            BlockStreamer(path, block_prefix="other.")

    def test_block_keys_unknown_idx_raises(self, tmp_path):
        sd = _block_state_dict("blocks.", num_blocks=2)
        path = _save_safetensors(tmp_path, "w", sd)
        streamer = BlockStreamer(path, block_prefix="blocks.")
        with pytest.raises(KeyError):
            streamer.block_keys(99)


class TestBlockStreamerBind:
    def test_bind_loads_correct_block(self, tmp_path):
        sd = _block_state_dict("b.", num_blocks=3, dim=4)
        path = _save_safetensors(tmp_path, "w", sd)
        streamer = BlockStreamer(path, block_prefix="b.")

        block = _Block(dim=4)
        streamer.bind(block, idx=1)
        # Block 1's linear.weight is all-ones * 2
        assert mx.array_equal(block.linear.weight, mx.ones((4, 4)) * 2).item()
        assert mx.array_equal(block.linear.bias, mx.ones((4,))).item()

    def test_bind_rebinds_different_block(self, tmp_path):
        sd = _block_state_dict("b.", num_blocks=3, dim=4)
        path = _save_safetensors(tmp_path, "w", sd)
        streamer = BlockStreamer(path, block_prefix="b.")

        block = _Block(dim=4)
        streamer.bind(block, idx=0)
        streamer.bind(block, idx=2)
        # Block 2's linear.weight is all-ones * 3
        assert mx.array_equal(block.linear.weight, mx.ones((4, 4)) * 3).item()

    def test_bind_evict_previous_then_rebind(self, tmp_path):
        """After evicting + cycling through all blocks, streamer re-mmaps."""
        sd = _block_state_dict("b.", num_blocks=3, dim=4)
        path = _save_safetensors(tmp_path, "w", sd)
        streamer = BlockStreamer(path, block_prefix="b.")

        block = _Block(dim=4)
        prev: int | None = None
        for idx in range(3):
            streamer.bind(block, idx=idx, evict_previous=prev)
            prev = idx
        # All blocks evicted by the last evict_previous (idx 2 was *bound*
        # last, so prev=2 evicts it on next iter — let's force re-mmap).
        streamer.bind(block, idx=0, evict_previous=prev)  # evicts 2; 0 already gone
        assert mx.array_equal(block.linear.weight, mx.ones((4, 4)) * 1).item()

    def test_bind_unknown_idx_raises(self, tmp_path):
        sd = _block_state_dict("b.", num_blocks=2)
        path = _save_safetensors(tmp_path, "w", sd)
        streamer = BlockStreamer(path, block_prefix="b.")
        block = _Block()
        with pytest.raises(KeyError, match="block 99"):
            streamer.bind(block, idx=99)


class TestBlockStreamerMultiFile:
    def test_merges_multiple_files(self, tmp_path):
        sd_a = {"b.0.linear.weight": mx.ones((4, 4)), "b.0.linear.bias": mx.zeros((4,))}
        sd_b = {"b.1.linear.weight": mx.ones((4, 4)) * 7, "b.1.linear.bias": mx.zeros((4,)) + 1}
        path_a = _save_safetensors(tmp_path, "a", sd_a)
        path_b = _save_safetensors(tmp_path, "b", sd_b)

        streamer = BlockStreamer([path_a, path_b], block_prefix="b.")
        assert streamer.block_count == 2

        block = _Block(dim=4)
        streamer.bind(block, idx=1)
        assert mx.array_equal(block.linear.weight, mx.ones((4, 4)) * 7).item()


class TestBlockStreamerLora:
    def test_lora_fuser_called_with_matching_sources(self, tmp_path):
        sd = _block_state_dict("b.", num_blocks=2, dim=4)
        weight_path = _save_safetensors(tmp_path, "w", sd)

        # Build a tiny LoRA file with A/B for block 0 only.
        lora_sd = {
            "b.0.linear.lora_A.weight": mx.ones((2, 4)),
            "b.0.linear.lora_B.weight": mx.ones((4, 2)),
        }
        lora_path = _save_safetensors(tmp_path, "lora", lora_sd)
        src = BlockLoraSource(lora_path, block_prefix="b.")
        assert src.has_block(0)
        assert not src.has_block(1)

        seen: list[tuple[int, int]] = []

        def fuser(
            weights: list[tuple[str, mx.array]],
            idx: int,
            sources: list[BlockLoraSource],
        ) -> list[tuple[str, mx.array]]:
            seen.append((idx, len(sources)))
            return weights

        streamer = BlockStreamer(weight_path, block_prefix="b.", lora_fuser=fuser)
        block = _Block(dim=4)
        streamer.bind(block, idx=0, lora_sources=[src])
        streamer.bind(block, idx=1, lora_sources=[src])
        # Fuser called only for block 0 (has_block filter removes block 1).
        assert seen == [(0, 1)]

    def test_lora_dict_shape(self, tmp_path):
        lora_sd = {
            "b.3.linear.lora_A.weight": mx.ones((2, 4)),
            "b.3.linear.lora_B.weight": mx.ones((4, 2)),
        }
        lora_path = _save_safetensors(tmp_path, "lora", lora_sd)
        src = BlockLoraSource(lora_path, block_prefix="b.")
        d = src.get_block_lora_dict(3)
        assert set(d.keys()) == {"linear.lora_A.weight", "linear.lora_B.weight"}

    def test_key_mapper_returning_none_drops_key(self, tmp_path):
        # Keys the mapper maps to None must be dropped even when the raw
        # key already matches the block prefix.
        lora_sd = {
            "b.0.linear.lora_A.weight": mx.ones((2, 4)),
            "b.0.linear.lora_B.weight": mx.ones((4, 2)),
            "b.1.linear.lora_A.weight": mx.ones((2, 4)),
            "b.1.linear.lora_B.weight": mx.ones((4, 2)),
        }
        lora_path = _save_safetensors(tmp_path, "lora", lora_sd)

        def drop_block_1(k: str) -> str | None:
            return None if k.startswith("b.1.") else k

        src = BlockLoraSource(lora_path, block_prefix="b.", key_mapper=drop_block_1)
        assert src.has_block(0)
        assert not src.has_block(1)
        assert src.get_block_lora_dict(1) == {}

    def test_key_mapper_returning_none_for_all_keys_yields_no_blocks(self, tmp_path):
        lora_sd = {
            "b.0.linear.lora_A.weight": mx.ones((2, 4)),
            "b.0.linear.lora_B.weight": mx.ones((4, 2)),
        }
        lora_path = _save_safetensors(tmp_path, "lora", lora_sd)
        src = BlockLoraSource(lora_path, block_prefix="b.", key_mapper=lambda _k: None)
        assert not src.has_block(0)
        assert src.get_block_lora_dict(0) == {}

    def test_key_mapper_remaps_lora_keys(self, tmp_path):
        # Raw safetensors uses an upstream naming; key_mapper rewrites
        # to the model's "b.<idx>." prefix.
        lora_sd = {
            "upstream.0.linear.lora_A.weight": mx.ones((2, 4)),
            "upstream.0.linear.lora_B.weight": mx.ones((4, 2)),
        }
        lora_path = _save_safetensors(tmp_path, "lora", lora_sd)

        def remap(k: str) -> str | None:
            return k.replace("upstream.", "b.") if k.startswith("upstream.") else None

        src = BlockLoraSource(lora_path, block_prefix="b.", key_mapper=remap)
        assert src.has_block(0)
        assert "linear.lora_A.weight" in src.get_block_lora_dict(0)


class TestBlockStreamerClose:
    def test_close_releases_state(self, tmp_path):
        sd = _block_state_dict("b.", num_blocks=2)
        path = _save_safetensors(tmp_path, "w", sd)
        streamer = BlockStreamer(path, block_prefix="b.")
        streamer.close()
        assert streamer.block_count == 0


class TestBlockLoraSourceClose:
    def test_close_makes_source_report_no_blocks(self, tmp_path):
        # Contract: "After this the source is unusable." — the implementation
        # degrades safely rather than raising: has_block goes False and
        # get_block_lora_dict returns {}, so a fuser sees no deltas.
        lora_sd = {
            "b.0.linear.lora_A.weight": mx.ones((2, 4)),
            "b.0.linear.lora_B.weight": mx.ones((4, 2)),
        }
        lora_path = _save_safetensors(tmp_path, "lora", lora_sd)
        src = BlockLoraSource(lora_path, block_prefix="b.")
        assert src.has_block(0)
        src.close()
        assert not src.has_block(0)
        assert src.get_block_lora_dict(0) == {}


class TestNonDictLoadRaises:
    """`.npy` paths make mx.load return an array, not a dict — both entrypoints must raise."""

    def test_block_streamer_raises_typeerror(self, tmp_path):
        path = str(tmp_path / "rogue.npy")
        mx.save(path, mx.ones((2, 2)))
        with pytest.raises(TypeError, match="expected dict from safetensors"):
            BlockStreamer(path, block_prefix="b.")

    def test_block_lora_source_raises_typeerror(self, tmp_path):
        path = str(tmp_path / "rogue.npy")
        mx.save(path, mx.ones((2, 2)))
        with pytest.raises(TypeError, match="expected dict from safetensors"):
            BlockLoraSource(path, block_prefix="b.")
