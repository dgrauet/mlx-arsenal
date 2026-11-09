"""ADR-0002 gate 3: the staged block-sparse kernel builds and matches dense attention.

CI installs the newest ``mlx`` that ``pyproject.toml`` allows, so an MLX release
that changes the private steel headers fails here. The scheduled
``mlx-main-kernel`` workflow runs this file against MLX main's headers
(``MLX_ARSENAL_STEEL_INCLUDE``).
"""

import shutil
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest

from mlx_arsenal.attention import _sol_kernel as sk
from mlx_arsenal.attention import block_sparse_attention

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

#: Max abs error against the reference, per dtype. Worst measured on M2 Pro: 2.4e-4, 2.0e-3, 1.0e-7.
ATOL = {mx.float16: 1e-3, mx.bfloat16: 8e-3, mx.float32: 1e-5}

#: The headers the kernel inlines from MLX 0.32.4 (and MLX main at c69303b). A change means:
#: re-check the kernel against the new headers, then update this list.
EXPECTED_HEADERS = {
    "steel/attn/attn.h",
    "steel/attn/loader.h",
    "steel/attn/mma.h",
    "steel/attn/params.h",
    "steel/attn/transforms.h",
    "steel/defines.h",
    "steel/gemm/params.h",
    "steel/utils.h",
    "steel/utils/integral_constant.h",
    "steel/utils/type_traits.h",
}


def _qkv(
    B: int, N: int, H: int, D: int, dtype: mx.Dtype, seed: int = 0
) -> tuple[mx.array, mx.array, mx.array]:
    """q, k, v as attention layers produce them: ``(B, N, H, D)`` transposed to ``(B, H, N, D)``."""
    kq, kk, kv = mx.random.split(mx.random.key(seed), 3)
    q, k, v = (
        mx.random.normal((B, N, H, D), key=key).astype(dtype).transpose(0, 2, 1, 3)
        for key in (kq, kk, kv)
    )
    return q, k, v


def _random_mask(B: int, H: int, nb: int, density: float, seed: int) -> mx.array:
    """Additive block mask: about ``density`` of the blocks routed, the diagonal always."""
    keep = mx.random.uniform(shape=(B, H, nb, nb), key=mx.random.key(seed)) < density
    keep = keep | mx.eye(nb, dtype=mx.bool_)
    return mx.where(keep, 0.0, -mx.inf)


def _max_err(out: mx.array, ref: mx.array) -> float:
    return cast(float, mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item())


class TestIncludeDir:
    def test_include_dir_has_the_headers(self):
        inc = sk.include_dir()
        assert (inc / "mlx/backend/metal/kernels/steel/attn/attn.h").is_file()

    def test_inlined_header_set(self):
        inc = sk.include_dir()
        base = inc / "mlx/backend/metal/kernels"
        got = {p.relative_to(base).as_posix() for p in sk.inlined_headers(inc)}
        assert got == EXPECTED_HEADERS

    def test_override_without_headers_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv(sk.INCLUDE_ENV, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="attn.h not found"):
            sk.include_dir()

    def test_override_is_used(self, monkeypatch, tmp_path):
        monkeypatch.delenv(sk.INCLUDE_ENV, raising=False)
        src = sk.include_dir() / "mlx/backend/metal/kernels/steel"
        shutil.copytree(src, tmp_path / "mlx/backend/metal/kernels/steel")
        monkeypatch.setenv(sk.INCLUDE_ENV, str(tmp_path))
        assert sk.include_dir() == Path(tmp_path)
        q, k, v = _qkv(1, 200, 2, 64, mx.float16)
        nb = 4
        out = sk.block_sparse_kernel(q, k, v, block_mask=mx.zeros((1, 1, nb, nb)), scale=0.125)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
        assert _max_err(out.transpose(0, 2, 1, 3), ref) < ATOL[mx.float16]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("D", [64, 128])
class TestParity:
    def test_all_routed_is_dense_sdpa(self, dtype, D):
        # 9 blocks, the last partial: every array input is in device memory, as in a real call.
        N = 8 * 64 + 3
        q, k, v = _qkv(1, N, 2, D, dtype)
        nb = -(-N // 64)
        out = sk.block_sparse_kernel(q, k, v, block_mask=mx.zeros((1, 1, nb, nb)), scale=D**-0.5)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
        assert out.shape == (1, N, 2, D)
        assert out.dtype == dtype
        assert _max_err(out.transpose(0, 2, 1, 3), ref) < ATOL[dtype]

    def test_skipped_blocks_match_dense_backend(self, dtype, D):
        # 20 blocks: two centroid tiles of 16, the second partial; per-batch and per-head routes.
        N = 20 * 64 - 5
        q, k, v = _qkv(2, N, 3, D, dtype, seed=1)
        mask = _random_mask(2, 3, 20, 0.3, seed=2)
        out = sk.block_sparse_kernel(q, k, v, block_mask=mask, scale=D**-0.5)
        ref = block_sparse_attention(q, k, v, block_mask=mask, scale=D**-0.5)
        assert _max_err(out.transpose(0, 2, 1, 3), ref) < ATOL[dtype]


def test_broadcast_mask_matches_full_mask():
    N, nb = 6 * 64, 6
    q, k, v = _qkv(2, N, 4, 64, mx.float16, seed=3)
    shared = _random_mask(1, 1, nb, 0.4, seed=4)
    out = sk.block_sparse_kernel(q, k, v, block_mask=shared, scale=0.125)
    full_mask = mx.broadcast_to(shared, (2, 4, nb, nb))
    full = sk.block_sparse_kernel(q, k, v, block_mask=full_mask, scale=0.125)
    ref = block_sparse_attention(q, k, v, block_mask=shared, scale=0.125)
    assert mx.array_equal(out, full).item()
    assert _max_err(out.transpose(0, 2, 1, 3), ref) < ATOL[mx.float16]


def test_skipping_changes_the_output():
    # Guards the parity tests against a kernel that ignores the route.
    N, nb = 6 * 64, 6
    q, k, v = _qkv(1, N, 2, 64, mx.float16, seed=5)
    dense = sk.block_sparse_kernel(q, k, v, block_mask=mx.zeros((1, 1, nb, nb)), scale=0.125)
    diag = mx.where(mx.eye(nb, dtype=mx.bool_), 0.0, -mx.inf)[None, None]
    sparse = sk.block_sparse_kernel(q, k, v, block_mask=diag, scale=0.125)
    assert _max_err(sparse, dense) > 10 * ATOL[mx.float16]
