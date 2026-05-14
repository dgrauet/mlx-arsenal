"""Tests for loader module."""

import mlx.core as mx
import pytest

from mlx_arsenal.loader import (
    ContentReplacement,
    KeyValueOperationResult,
    SafetensorsStateDictLoader,
    SDOps,
    StateDict,
    read_safetensors_metadata,
)


def _save(
    tmp_path, name: str, sd: dict[str, mx.array], metadata: dict[str, str] | None = None
) -> str:
    path = str(tmp_path / name)
    if metadata is not None:
        mx.save_safetensors(path, sd, metadata=metadata)
    else:
        mx.save_safetensors(path, sd)
    return path if path.endswith(".safetensors") else path + ".safetensors"


# ---------------------------------------------------------------------------
# SDOps
# ---------------------------------------------------------------------------


class TestSDOpsMatching:
    def test_default_chain_drops_all_keys(self):
        # No ContentMatching in the chain → nothing passes.
        ops = SDOps("empty")
        assert ops.apply_to_key("transformer.weight") is None

    def test_open_matcher_accepts_all(self):
        ops = SDOps("open").with_matching()  # prefix="" suffix=""
        assert ops.apply_to_key("anything") == "anything"

    def test_prefix_filter(self):
        ops = SDOps("p").with_matching(prefix="model.")
        assert ops.apply_to_key("model.linear.weight") == "model.linear.weight"
        assert ops.apply_to_key("other.linear.weight") is None

    def test_suffix_filter(self):
        ops = SDOps("s").with_matching(suffix=".weight")
        assert ops.apply_to_key("a.weight") == "a.weight"
        assert ops.apply_to_key("a.bias") is None

    def test_multiple_matchers_are_or(self):
        ops = SDOps("any").with_matching(prefix="enc.").with_matching(prefix="dec.")
        assert ops.apply_to_key("enc.0.weight") == "enc.0.weight"
        assert ops.apply_to_key("dec.0.weight") == "dec.0.weight"
        assert ops.apply_to_key("other.weight") is None


class TestSDOpsReplacement:
    def test_single_replacement(self):
        ops = SDOps("r").with_matching().with_replacement("diffusion_model.", "")
        assert ops.apply_to_key("diffusion_model.to_q.weight") == "to_q.weight"

    def test_chained_replacements_apply_in_order(self):
        ops = (
            SDOps("c")
            .with_matching()
            .with_replacement("diffusion_model.", "")
            .with_replacement(".to_out.0.", ".to_out.")
        )
        assert ops.apply_to_key("diffusion_model.attn.to_out.0.weight") == "attn.to_out.weight"

    def test_replacement_only_when_substring_present(self):
        ops = SDOps("r").with_matching().with_replacement("foo", "bar")
        # Key without 'foo' is unchanged.
        assert ops.apply_to_key("baz.weight") == "baz.weight"


class TestSDOpsAllowedKeys:
    def test_allowed_keys_filters_post_replacement(self):
        ops = (
            SDOps("a")
            .with_matching()
            .with_replacement("old.", "new.")
            .with_additional_allowed_keys(frozenset({"new.weight"}))
        )
        assert ops.apply_to_key("old.weight") == "new.weight"
        # "new.bias" not in allowed set
        assert ops.apply_to_key("old.bias") is None

    def test_additional_allowed_keys_unions(self):
        ops = (
            SDOps("u")
            .with_matching()
            .with_additional_allowed_keys(frozenset({"a"}))
            .with_additional_allowed_keys(frozenset({"b"}))
        )
        assert ops.apply_to_key("a") == "a"
        assert ops.apply_to_key("b") == "b"
        assert ops.apply_to_key("c") is None


class TestSDOpsImmutability:
    def test_with_replacement_returns_new_instance(self):
        a = SDOps("a").with_matching()
        b = a.with_replacement("x", "y")
        assert a is not b
        assert a.mapping != b.mapping
        # Original unchanged
        assert all(not isinstance(op, ContentReplacement) for op in a.mapping)

    def test_uses_frozen_dataclass(self):
        ops = SDOps("z")
        with pytest.raises((AttributeError, Exception)):
            ops.name = "renamed"  # ty: ignore[invalid-assignment]


class TestSDOpsKeyValueOperation:
    def test_no_kv_op_passes_through(self):
        ops = SDOps("x").with_matching()
        arr = mx.ones((2, 2))
        pairs = ops.apply_to_key_value("k", arr)
        assert pairs == [KeyValueOperationResult("k", arr)]

    def test_kv_op_splits_one_key_into_many(self):
        """A kv operation can return multiple results — e.g. fused QKV split."""

        def split_qkv(tensor_key: str, tensor_value: mx.array) -> list[KeyValueOperationResult]:
            base = tensor_key.replace(".qkv.weight", "")
            q, k, v = mx.split(tensor_value, 3, axis=0)
            return [
                KeyValueOperationResult(f"{base}.q.weight", q),
                KeyValueOperationResult(f"{base}.k.weight", k),
                KeyValueOperationResult(f"{base}.v.weight", v),
            ]

        ops = SDOps("split").with_matching().with_kv_operation(split_qkv, key_suffix=".qkv.weight")
        fused = mx.ones((6, 4))
        pairs = ops.apply_to_key_value("attn.qkv.weight", fused)
        assert len(pairs) == 3
        names = [p.new_key for p in pairs]
        assert names == ["attn.q.weight", "attn.k.weight", "attn.v.weight"]
        for _, arr in pairs:
            assert arr.shape == (2, 4)

    def test_kv_op_only_fires_when_matcher_matches(self):
        def double(tensor_key: str, tensor_value: mx.array) -> list[KeyValueOperationResult]:
            return [KeyValueOperationResult(tensor_key, tensor_value * 2)]

        ops = SDOps("d").with_matching().with_kv_operation(double, key_suffix=".bias")
        # Suffix matches → operation runs.
        arr = mx.ones((4,))
        pairs = ops.apply_to_key_value("a.bias", arr)
        assert mx.allclose(pairs[0].new_value, mx.ones((4,)) * 2).item()
        # Suffix doesn't match → pass-through.
        pairs = ops.apply_to_key_value("a.weight", arr)
        assert mx.allclose(pairs[0].new_value, arr).item()


# ---------------------------------------------------------------------------
# SafetensorsStateDictLoader
# ---------------------------------------------------------------------------


class TestSafetensorsLoader:
    def test_load_without_sd_ops_returns_all_keys(self, tmp_path):
        path = _save(tmp_path, "w", {"a": mx.ones((2, 2)), "b": mx.zeros((4,))})
        sd = SafetensorsStateDictLoader().load(path)
        assert set(sd.sd) == {"a", "b"}
        assert sd.size == sd.sd["a"].nbytes + sd.sd["b"].nbytes
        assert mx.float32 in sd.dtype

    def test_load_with_sd_ops_renames_keys(self, tmp_path):
        path = _save(tmp_path, "w", {"diffusion_model.attn.weight": mx.ones((2, 2))})
        ops = SDOps("p").with_matching().with_replacement("diffusion_model.", "")
        sd = SafetensorsStateDictLoader().load(path, sd_ops=ops)
        assert set(sd.sd) == {"attn.weight"}

    def test_load_drops_unmatched_keys(self, tmp_path):
        path = _save(tmp_path, "w", {"keep.x": mx.ones((2,)), "drop.y": mx.zeros((2,))})
        ops = SDOps("p").with_matching(prefix="keep.")
        sd = SafetensorsStateDictLoader().load(path, sd_ops=ops)
        assert set(sd.sd) == {"keep.x"}

    def test_load_multiple_shards_merges(self, tmp_path):
        p1 = _save(tmp_path, "s1", {"a": mx.ones((2, 2))})
        p2 = _save(tmp_path, "s2", {"b": mx.zeros((4,))})
        sd = SafetensorsStateDictLoader().load([p1, p2])
        assert set(sd.sd) == {"a", "b"}

    def test_size_and_dtype_accumulate(self, tmp_path):
        path = _save(
            tmp_path,
            "w",
            {"a": mx.ones((2, 2), dtype=mx.float32), "b": mx.zeros((4,), dtype=mx.float16)},
        )
        sd = SafetensorsStateDictLoader().load(path)
        assert sd.dtype == {mx.float32, mx.float16}
        assert sd.footprint() == sd.size

    def test_kv_op_splits_at_load_time(self, tmp_path):
        """End-to-end: fused QKV split during load."""
        path = _save(tmp_path, "w", {"attn.qkv.weight": mx.ones((6, 4))})

        def split_qkv(tensor_key: str, tensor_value: mx.array) -> list[KeyValueOperationResult]:
            base = tensor_key.replace(".qkv.weight", "")
            q, k, v = mx.split(tensor_value, 3, axis=0)
            return [
                KeyValueOperationResult(f"{base}.q.weight", q),
                KeyValueOperationResult(f"{base}.k.weight", k),
                KeyValueOperationResult(f"{base}.v.weight", v),
            ]

        ops = SDOps("split").with_matching().with_kv_operation(split_qkv, key_suffix=".qkv.weight")
        sd = SafetensorsStateDictLoader().load(path, sd_ops=ops)
        assert set(sd.sd) == {"attn.q.weight", "attn.k.weight", "attn.v.weight"}


# ---------------------------------------------------------------------------
# read_safetensors_metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_no_metadata_returns_empty(self, tmp_path):
        path = _save(tmp_path, "w", {"a": mx.ones((2,))})
        assert read_safetensors_metadata(path) == {}

    def test_round_trip_metadata(self, tmp_path):
        path = _save(
            tmp_path, "w", {"a": mx.ones((2,))}, metadata={"format": "mlx", "version": "1.0"}
        )
        meta = read_safetensors_metadata(path)
        assert meta == {"format": "mlx", "version": "1.0"}


# ---------------------------------------------------------------------------
# StateDict
# ---------------------------------------------------------------------------


class TestStateDict:
    def test_footprint(self):
        sd = StateDict(sd={"a": mx.ones((4, 4))}, size=64, dtype={mx.float32})
        assert sd.footprint() == 64

    def test_immutability(self):
        sd = StateDict(sd={"a": mx.ones((4, 4))}, size=64, dtype={mx.float32})
        with pytest.raises((AttributeError, Exception)):
            sd.size = 999  # ty: ignore[invalid-assignment]
