"""Block-sparse Metal attention kernel, staged for ADR-0002 phase 2 (private, not dispatched).

:func:`~mlx_arsenal.attention.block_sparse_attention` does not call this module:
:func:`~mlx_arsenal.attention.block_sparse_kernel_available` stays ``False``
until all four ADR-0002 gates pass. The module is here so that CI can build it
against the installed ``mlx`` (gate 3) and fail when an MLX release changes the
private headers it compiles against.

The kernel is MLX's steel flash-attention loop
(``steel/attn/kernels/steel_attention.h``, MIT, Copyright Apple; BQ=32, BK=16,
4 simdgroups, the tile shape MLX uses for head dim 128 on GPUs without neural
accelerators) run over the routed key blocks only, then over the centroids of
the skipped blocks. A skipped block enters the softmax once, as its key
centroid with a ``log(block length)`` bias and its value mean:
``exp(s + ln L) * mean(V) = exp(s) * sum(V)``. This is the compensation of
Sol-Attn (arXiv 2607.24027, Eqs. 8-10) and what the dense backend of
``block_sparse_attention(compensate=True)`` computes. The code is adapted from
ltx-2-mlx (MIT); it contains no code from NVlabs/Sana.

The kernel inlines ``steel/attn/attn.h`` and its local includes, read from the
installed ``mlx`` package (``mx.fast.metal_kernel`` takes one source string).
``MLX_ARSENAL_STEEL_INCLUDE`` points the build at another include root, which
the scheduled CI job uses to build against MLX main.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import mlx
import mlx.core as mx

#: Tokens per routing block; the kernel's indexing assumes it.
BLOCK = 64
#: Environment variable naming an include root to build against instead of the installed ``mlx``.
INCLUDE_ENV = "MLX_ARSENAL_STEEL_INCLUDE"
_ATTN_HEADER = "mlx/backend/metal/kernels/steel/attn/attn.h"

_OPS = r"""
struct MaxOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); } };
struct SumOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; } };
struct MulOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; } };
struct ExpSubOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); } };
struct DivOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x / y; } };
"""

# One key tile through MLX's steel score / softmax / output update (steel_attention.h,
# WN = 1, no mask / causal / sinks). CENT = the tile holds centroid keys of skipped
# blocks: centroids of routed or out-of-range blocks are masked and the others get the
# log2(length) bias; otherwise exact keys, masked past `rem` in a partial last block.
_TILE = r"""
template <bool CENT, typename T, int BK, int BD, int LDQ_tgp, int LDK_tgp, int LDV_tgp,
          typename KBlockLoader, typename VBlockLoader, typename RP, typename CBP, typename QT, typename KT, typename ST,
          typename VT, typename OT>
METAL_FUNC void sol_tile(
    const device T* kp, const device T* vp, int kld, int vld, int rem, int c_first, int nb,
    RP R, CBP CB, float scale, threadgroup T* Qs, threadgroup T* KVs,
    short Qs_offset, short Ks_offset, short Vs_offset, short sn, uint simd_group_id, uint simd_lane_id,
    thread QT& Qtile, thread KT& Ktile, thread ST& Stile, thread VT& Vtile, thread OT& Otile,
    thread float* max_score, thread float* sum_score) {
  constexpr short kFragSize = 8;
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  using MMAFrag_acc_t = BaseMMAFrag<float, kFragSize, kFragSize>;
  constexpr short kRowsPT = ST::kRowsPerThread;
  constexpr float neg_inf = Limits<float>::finite_min;
  KBlockLoader loader_k(kp, kld, KVs, simd_group_id, simd_lane_id);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (rem < BK) { loader_k.load_safe(short2(BD, rem)); } else { loader_k.load_unsafe(); }
  Stile.clear();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for (short dd = 0; dd < TD; dd++) {
    simdgroup_barrier(mem_flags::mem_none);
    Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * kFragSize]);
    Ktile.template load<T, 1, 1, LDK_tgp, 1>(&KVs[Ks_offset + dd * kFragSize * LDK_tgp]);
    simdgroup_barrier(mem_flags::mem_none);
    tile_matmad(Stile, Qtile, Ktile, Stile);
  }
  STEEL_PRAGMA_UNROLL
  for (short ii = 0; ii < ST::kElemsPerTile; ii++) { Stile.elems()[ii] *= scale; }
  if (CENT) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
        const int c = c_first + sn + j * kFragSize + jj;
        if (c >= nb || R[c]) { Stile.frag_at(0, j)[jj] = neg_inf; } else { Stile.frag_at(0, j)[jj] += CB[c]; }
      }
    }
  } else if (rem < BK) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
        if (sn + j * kFragSize + jj >= rem) { Stile.frag_at(0, j)[jj] = neg_inf; }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  VBlockLoader loader_v(vp, vld, KVs, simd_group_id, simd_lane_id);
  if (rem < BK) { loader_v.load_safe(short2(BD, rem)); } else { loader_v.load_unsafe(); }

  float new_max[kRowsPT];
  float factor[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
  Stile.template row_reduce<MaxOp>(new_max);
  Stile.template row_bin_op<ExpSubOp>(new_max);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { factor[i] = fast::exp2(max_score[i] - new_max[i]); max_score[i] = new_max[i]; }
  float sum_score_tmp[kRowsPT] = {0};
  Stile.template row_reduce<SumOp>(sum_score_tmp);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i]; }
  Otile.template row_bin_op<MulOp>(factor);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TD; id++) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      simdgroup_barrier(mem_flags::mem_none);
      Vtile.template load<T, 1, 1, LDV_tgp, 1>(&KVs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
      simdgroup_barrier(mem_flags::mem_none);
      MMAFrag_acc_t::mma(Otile.frag_at(0, id), Stile.frag_at(0, ik), Vtile.frag_at(0, 0), Otile.frag_at(0, id));
    }
  }
}

// CBP / RP: MLX passes inputs with fewer than 8 elements in the constant address space, the others in device memory.
template <typename T, int BQ, int BK, int BD, int WM, typename CBP, typename RP>
METAL_FUNC void sol_attention_impl(
    const device T* Q, const device T* K, const device T* V, const device T* KC, const device T* VC,
    CBP CB, RP R, device T* O,
    const int64_t qs_b, const int64_t qs_h, const int64_t qs_t, const int64_t ks_b, const int64_t ks_h,
    const int64_t ks_t, const int64_t vs_b, const int64_t vs_h, const int64_t vs_t,
    const int H, const int N, const int nb, const float softmax_scale,
    threadgroup T* Qs, threadgroup T* KVs, uint simd_lane_id, uint simd_group_id, uint3 tid) {
  constexpr short kFragSize = 8;
  constexpr short pad = 16 / sizeof(T);
  constexpr short LDQ_tgp = BD + pad;
  constexpr short LDK_tgp = BK + pad;
  constexpr short LDV_tgp = BD + pad;
  constexpr int TGP = WM * 32;
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, TGP>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, TGP>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, TGP>;
  using MMAFrag_acc_t = BaseMMAFrag<float, kFragSize, kFragSize>;
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  static_assert(BQ == WM * kFragSize, "one simdgroup matrix row per simdgroup");

  const int q0 = int(tid.x) * BQ;
  const int h = int(tid.y);
  const int b = int(tid.z);
  const ulong bh = ulong(b) * H + h;
  Q += b * qs_b + h * qs_h + q0 * qs_t;
  K += b * ks_b + h * ks_h;
  V += b * vs_b + h * vs_h;
  KC += bh * nb * BD;
  VC += bh * nb * BD;
  R += (bh * nb + q0 / 64) * nb;
  O += ((ulong(b) * N + q0) * H + h) * BD;
  const int q_rem = N - q0;

  QBlockLoader loader_q(Q, int(qs_t), Qs, simd_group_id, simd_lane_id);
  const float scale = softmax_scale * M_LOG2E_F;

  MMATile<float, 1, 1, MMAFrag_acc_t> Qtile;
  MMATile<float, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<float, 1, TK, MMAFrag_acc_t> Stile;
  MMATile<float, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<float, 1, TD, MMAFrag_acc_t> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * simd_group_id;
  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (q_rem < BQ) { loader_q.load_safe(short2(BD, q_rem)); } else { loader_q.load_unsafe(); }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  float max_score[kRowsPT];
  float sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { max_score[i] = Limits<float>::finite_min; }

#define SOL_TILE(CENT, KP, VP, KLD, VLD, REM, CF) \
  sol_tile<CENT, T, BK, BD, LDQ_tgp, LDK_tgp, LDV_tgp, KBlockLoader, VBlockLoader>( \
      KP, VP, KLD, VLD, REM, CF, nb, R, CB, scale, Qs, KVs, Qs_offset, Ks_offset, Vs_offset, sn, \
      simd_group_id, simd_lane_id, Qtile, Ktile, Stile, Vtile, Otile, max_score, sum_score)
  // Routed blocks, exact. Full blocks pass a literal BK (unchecked loads, as the dense kernel); token offsets fit in
  // int. The loop over the 4 tiles of a block is not unrolled: that measured 5x slower (code size). The diagonal block
  // must be routed, so every row has a real maximum before the centroid tiles come in.
  const int kst = int(ks_t), vst = int(vs_t);
  for (int kb = 0; kb < nb; kb++) {
    if (!R[kb]) { continue; }
    const int k0 = kb * 64;
    if (k0 + 64 <= N) {
      _Pragma("clang loop unroll(disable)")
      for (int t = 0; t < 64; t += BK) { SOL_TILE(false, K + (k0 + t) * kst, V + (k0 + t) * vst, kst, vst, BK, 0); }
    } else {
      for (int t = 0; t < N - k0; t += BK) {
        SOL_TILE(false, K + (k0 + t) * kst, V + (k0 + t) * vst, kst, vst, N - k0 - t, 0);
      }
    }
  }
  // Skipped blocks, as centroid keys; a centroid tile whose blocks are all routed is skipped.
  for (int c0 = 0; c0 < nb; c0 += BK) {
    bool any = false;
    for (int c = c0; c < min(c0 + BK, nb); c++) { any = any || !R[c]; }
    if (!any) { continue; }
    SOL_TILE(true, KC + c0 * BD, VC + c0 * BD, BD, BD, nb - c0, c0);
  }
#undef SOL_TILE

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);
  O += (tm + sm) * H * BD + sn;
  if (q_rem < BQ) {
    const short2 dims = short2(BD - sn, q_rem - (tm + sm));
    if (dims.x <= 0 || dims.y <= 0) { return; }
    Otile.template store_safe<T, 1, 1>(O, H * BD, dims);
  } else {
    Otile.template store<T, 1, 1>(O, H * BD);
  }
}
"""

_ATTN_BODY = r"""
  constexpr short pad = 16 / sizeof(T);
  constexpr short m0 = (BK + pad) * BD;
  constexpr short m1 = BK * (BD + pad);
  threadgroup T Q_smem[BQ * (BD + pad)];
  threadgroup T KV_smem[m0 > m1 ? m0 : m1];
  sol_attention_impl<T, BQ, BK, BD, WM>(
      q, k, v, kc, vc, cb, route, out,
      q_strides[0], q_strides[1], q_strides[2], k_strides[0], k_strides[1], k_strides[2],
      v_strides[0], v_strides[1], v_strides[2],
      ip[0], ip[1], ip[2], sc[0], Q_smem, KV_smem,
      thread_index_in_simdgroup, simdgroup_index_in_threadgroup, threadgroup_position_in_grid);
"""

# One thread per (batch * head, block, dim): float32 sums of k and v over the
# 64 tokens of the block.
_SUM_BODY = r"""
  const uint H = ip[0], N = ip[1], nb = ip[2];
  const uint gid = thread_position_in_grid.x;
  const uint d = gid % BD;
  const uint c = (gid / BD) % nb;
  const uint bh = gid / (BD * nb);
  const uint b = bh / H, h = bh % H;
  const uint t0 = c * 64, t1 = min(t0 + 64, N);
  float sk = 0.0f, sv = 0.0f;
  for (uint t = t0; t < t1; t++) {
    sk += float(k[b * k_strides[0] + h * k_strides[1] + t * k_strides[2] + d]);
    sv += float(v[b * v_strides[0] + h * v_strides[1] + t * v_strides[2] + d]);
  }
  ks[gid] = sk; vs[gid] = sv;
"""

# mlx types `fast.metal_kernel` as returning `object`; the kernels are callables.
# Built kernels per include root. mlx types `fast.metal_kernel` as returning `object`.
_KERNELS: dict[Path, dict[str, Any]] = {}


def include_dir() -> Path:
    """The include root the kernel builds against.

    ``$MLX_ARSENAL_STEEL_INCLUDE`` when set, else the ``include`` directory of
    the installed ``mlx`` package.

    Raises:
        FileNotFoundError: If the root lacks ``steel/attn/attn.h``.
    """
    override = os.environ.get(INCLUDE_ENV)
    roots = [Path(override)] if override else [Path(p) / "include" for p in mlx.__path__]
    for inc in roots:
        if (inc / _ATTN_HEADER).exists():
            return inc
    raise FileNotFoundError(f"{_ATTN_HEADER} not found under {', '.join(map(str, roots))}")


def _inline(inc: Path, path: Path, seen: set[Path]) -> str:
    """The header with its local includes pasted in, each one once."""
    out = []
    for line in path.read_text().splitlines():
        m = re.match(r'\s*#include\s+"(.+)"', line)
        if m:
            p = inc / m.group(1)
            if p not in seen:
                seen.add(p)
                out.append(_inline(inc, p, seen))
            continue
        if line.strip() == "#pragma once":
            continue
        out.append(line)
    return "\n".join(out)


def inlined_headers(inc: Path) -> list[Path]:
    """The steel headers the kernel inlines from ``inc``, in inclusion order."""
    seen: set[Path] = set()
    _inline(inc, inc / _ATTN_HEADER, seen)
    return [inc / _ATTN_HEADER, *sorted(seen - {inc / _ATTN_HEADER})]


def _kernels() -> dict[str, Any]:
    inc = include_dir()
    if inc not in _KERNELS:
        header = (
            _inline(inc, inc / _ATTN_HEADER, set())
            + "\nusing namespace mlx::steel;\n"
            + _OPS
            + _TILE
        )
        kernels: dict[str, Any] = {
            "attn": mx.fast.metal_kernel(
                name="arsenal_block_sparse_attention",
                input_names=["q", "k", "v", "kc", "vc", "cb", "route", "ip", "sc"],
                output_names=["out"],
                header=header,
                source=_ATTN_BODY,
                ensure_row_contiguous=False,
            ),
            "sum": mx.fast.metal_kernel(
                name="arsenal_block_sums",
                input_names=["k", "v", "ip"],
                output_names=["ks", "vs"],
                source=_SUM_BODY,
                ensure_row_contiguous=False,
            ),
        }
        _KERNELS[inc] = kernels
    return _KERNELS[inc]


def block_sparse_kernel(
    q: mx.array, k: mx.array, v: mx.array, *, block_mask: mx.array, scale: float
) -> mx.array:
    """Compensated block-sparse attention on the Metal kernel.

    Computes what ``block_sparse_attention(q, k, v, block_mask=block_mask,
    scale=scale)`` computes with ``block_size=64`` and ``compensate=True``.
    No input checks: the caller guarantees the contract below.

    Args:
        q, k, v: ``(B, H, N, D)`` float16, bfloat16 or float32, the head dim
            contiguous and any other strides (e.g. transposed out of
            ``(B, N, H, D)``). ``D`` is a multiple of 8.
        block_mask: ``(1|B, 1|H, nb, nb)`` additive mask, ``nb = ceil(N /
            64)``: ``0`` routes the key block exactly, anything else replaces
            it by its centroid. The diagonal must be routed.
        scale: Softmax scale.

    Returns:
        ``(B, N, H, D)`` in ``q.dtype``; ``.transpose(0, 2, 1, 3)`` gives the
        ``(B, H, N, D)`` layout of ``block_sparse_attention``.
    """
    B, H, N, D = q.shape
    nb = -(-N // BLOCK)
    kernels = _kernels()
    ip = mx.array([H, N, nb], dtype=mx.int32)
    ks, vs = kernels["sum"](
        inputs=[k, v, ip],
        template=[("BD", D)],
        grid=(B * H * nb * D, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, H, nb, D)] * 2,
        output_dtypes=[mx.float32] * 2,
    )
    lengths = mx.minimum(BLOCK, N - mx.arange(nb) * BLOCK).astype(mx.float32)
    kc = (ks / lengths[:, None]).astype(k.dtype)
    vmean = (vs / lengths[:, None]).astype(v.dtype)
    route = mx.contiguous(mx.broadcast_to(mx.equal(block_mask, 0), (B, H, nb, nb)).astype(mx.uint8))
    bq, bk, wm = 32, 16, 4
    return kernels["attn"](
        inputs=[
            q,
            k,
            v,
            kc,
            vmean,
            mx.log2(lengths),
            route,
            ip,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("T", q.dtype), ("BQ", bq), ("BK", bk), ("BD", D), ("WM", wm)],
        grid=(-(-N // bq) * 32, H * wm, B),
        threadgroup=(32, wm, 1),
        output_shapes=[(B, N, H, D)],
        output_dtypes=[q.dtype],
    )[0]
