"""State-dict key renaming and matching operations.

A common need when porting models is to rewrite upstream weight-key
naming (diffusers ``net.0.proj``, ComfyUI ``diffusion_model.*``,
Megatron ``linear_fc1``) into the local MLX module's parameter tree.
:class:`SDOps` chains immutable rename / match / transform operations
that are applied in order to each key (and optionally to each
key/value pair) at load time.

Three operation kinds compose:

- :class:`ContentReplacement` — substring rename (``str.replace``).
- :class:`ContentMatching` — prefix/suffix filter; keys that don't
  match at least one such filter are dropped.
- :class:`SDKeyValueOperation` — full key/value transform, e.g. to
  split a fused QKV tensor into three keys.

Build the chain with the fluent ``with_*`` methods on :class:`SDOps`,
then pass the resulting :class:`SDOps` to a loader's ``load`` method::

    ops = (
        SDOps("comfy_to_mlx")
        .with_matching()                            # accept all keys
        .with_replacement("diffusion_model.", "")
        .with_replacement(".to_out.0.", ".to_out.")
    )

    state = SafetensorsStateDictLoader().load(path, sd_ops=ops)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol

import mlx.core as mx

__all__ = [
    "ContentMatching",
    "ContentReplacement",
    "KeyValueOperation",
    "KeyValueOperationResult",
    "SDKeyValueOperation",
    "SDOps",
]


@dataclass(frozen=True, slots=True)
class ContentReplacement:
    """A substring replacement applied to a state-dict key."""

    content: str
    replacement: str


@dataclass(frozen=True, slots=True)
class ContentMatching:
    """Prefix/suffix matcher gating which keys :class:`SDOps` lets through.

    Defaults to ``prefix=""`` and ``suffix=""`` which matches everything.
    A key passes :meth:`SDOps.apply_to_key` only if it matches *at least
    one* :class:`ContentMatching` in the chain.
    """

    prefix: str = ""
    suffix: str = ""


class KeyValueOperationResult(NamedTuple):
    """Output of a key-value transform: a (new key, new value) pair.

    Operations may return multiple results (e.g. splitting a fused
    QKV tensor produces three results).
    """

    new_key: str
    new_value: mx.array


class KeyValueOperation(Protocol):
    """Callable protocol for arbitrary key/value transforms."""

    def __call__(
        self,
        tensor_key: str,
        tensor_value: mx.array,
    ) -> list[KeyValueOperationResult]: ...


@dataclass(frozen=True, slots=True)
class SDKeyValueOperation:
    """A :class:`KeyValueOperation` gated by a :class:`ContentMatching`."""

    key_matcher: ContentMatching
    kv_operation: KeyValueOperation


@dataclass(frozen=True, slots=True)
class SDOps:
    """Immutable chain of state-dict operations.

    Every fluent method returns a new :class:`SDOps`; the original is
    never mutated. This makes chains safe to share across loaders and
    safe to compose into reusable presets.

    Attributes:
        name: Human-readable label for the chain (used for debugging
            and serialization).
        mapping: Ordered tuple of operations to apply.
        allowed_keys: If set, keys not in this frozenset (after all
            replacements have been applied) are dropped.
    """

    name: str
    mapping: tuple[ContentReplacement | ContentMatching | SDKeyValueOperation, ...] = ()
    allowed_keys: frozenset[str] | None = None

    def with_replacement(self, content: str, replacement: str) -> SDOps:
        """Append a substring replacement to the chain."""
        return replace(self, mapping=(*self.mapping, ContentReplacement(content, replacement)))

    def with_matching(self, prefix: str = "", suffix: str = "") -> SDOps:
        """Append a prefix/suffix filter to the chain.

        Defaults to ``prefix=""`` / ``suffix=""``, which matches every
        key. Use empty defaults when you want to allow every key
        through and rely on later replacements / allowed-keys to gate
        them.
        """
        return replace(self, mapping=(*self.mapping, ContentMatching(prefix, suffix)))

    def with_additional_allowed_keys(self, keys: frozenset[str]) -> SDOps:
        """Restrict (or further restrict) the post-replacement key set.

        If :attr:`allowed_keys` is already set, the two sets are
        union'd — additive, never subtractive.
        """
        merged = (
            frozenset(keys) | self.allowed_keys
            if self.allowed_keys is not None
            else frozenset(keys)
        )
        return replace(self, allowed_keys=merged)

    def with_kv_operation(
        self,
        operation: KeyValueOperation,
        key_prefix: str = "",
        key_suffix: str = "",
    ) -> SDOps:
        """Append a key/value transform gated by an optional matcher."""
        sd_kv = SDKeyValueOperation(ContentMatching(key_prefix, key_suffix), operation)
        return replace(self, mapping=(*self.mapping, sd_kv))

    def apply_to_key(self, key: str) -> str | None:
        """Apply matching + replacements. Returns ``None`` if the key is dropped.

        A key is dropped when (a) no :class:`ContentMatching` in the
        chain accepts it, or (b) the post-replacement key is not in
        :attr:`allowed_keys` (when set).
        """
        matchers = [c for c in self.mapping if isinstance(c, ContentMatching)]
        if not any(key.startswith(m.prefix) and key.endswith(m.suffix) for m in matchers):
            return None

        for op in self.mapping:
            if isinstance(op, ContentReplacement) and op.content in key:
                key = key.replace(op.content, op.replacement)

        if self.allowed_keys is not None and key not in self.allowed_keys:
            return None

        return key

    def apply_to_key_value(self, key: str, value: mx.array) -> list[KeyValueOperationResult]:
        """Apply the first matching :class:`SDKeyValueOperation`, if any.

        Returns a single ``[(key, value)]`` if no kv operation matches
        — i.e. the value is passed through untouched. Only the first
        matching operation is invoked; design the chain accordingly.
        """
        for op in self.mapping:
            if not isinstance(op, SDKeyValueOperation):
                continue
            m = op.key_matcher
            if key.startswith(m.prefix) and key.endswith(m.suffix):
                return op.kv_operation(key, value)
        return [KeyValueOperationResult(key, value)]
