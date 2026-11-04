# ADR-0002 gate log

Status of the four gates that [ADR-0002](../adr/0002-block-sparse-attention-kernel.md)
sets before the ltx-2-mlx Sol-Attn Metal kernel can move behind
`block_sparse_attention`. Each entry is dated; the kernel stays in the port
until all four pass.

## Summary (2026-11-02)

| Gate | Status |
|---|---|
| 1. Hardware and shapes | Partial: M1 Max and M2 Pro pass at head dim 64 and 128, float16 and bfloat16. No chip with GPU neural accelerators (M5 family) is available. |
| 2. Second consumer | In progress: ERNIE-Image at 2048² runs through the arsenal router and the kernel, 1.42× per step. The image-quality run against dense is still to do. |
| 3. MLX coupling | Not yet a CI test, but the risk is lower than the October review assumed (see below). |
| 4. Licence | Closed: the routing follows the Sol-Attn paper, not the unlicensed `sol-engine` code. |

## Gate 1: hardware and shapes

**Chips.** M1 Max (`applegpu_g13s`, the original ltx-2-mlx measurements) and
M2 Pro (`applegpu_g14s`, 2026-11-02, MLX 0.32.4). Neither has GPU neural
accelerators, so MLX's dense SDPA there uses the same pre-NAX steel tiles as
the kernel. The NAX case (M5 family) stays open: no such machine is
available. The 2027-04 review decides, as the ADR states.

**Parity on M2 Pro.** Kernel against the dense backend of
`block_sparse_attention` with the same route (`centroid_threshold_block_mask`,
`τ = 1`, `band = 1`), random inputs, 4 heads:

| dtype | head dim | 4096 tokens | 4000 tokens (partial block) |
|---|---|---|---|
| float16 | 64 | 5.7e-5 | 6.1e-5 |
| float16 | 128 | 5.2e-5 | 6.1e-5 |
| bfloat16 | 64 | 4.7e-4 | 4.8e-4 |
| bfloat16 | 128 | 3.8e-4 | 4.8e-4 |

(max absolute difference; route density 0.20). At longer lengths the
kernel matches a float64 reference to about 6e-5, which exposed a bug in
the dense backend itself (gate 3 below).

**Speed on M2 Pro.** 16 heads, random inputs, kernel time including the
block summaries and the route:

| dtype | head dim | 2048 | 4096 | 8192 | 16384 | 32768 tokens |
|---|---|---|---|---|---|---|
| float16 | 128, `τ = 1.0` (density 0.24 → 0.16) | 2.83× | 3.80× | 4.47× | 4.35× | 5.01× |
| float16 | 128, `τ = 1.5` (density 0.15 → 0.07) | 3.67× | 5.59× | 7.31× | 8.85× | 9.72× |
| bfloat16 | 128, `τ = 1.0` | 3.07× | 4.02× | 4.76× | 5.16× | 4.90× |
| float16 | 64, `τ = 1.0` | 2.51× | 3.38× | 4.05× | 4.47× | 4.91× |

Random inputs route sparsely; real attention keeps more blocks, so the
useful number is the speed at a given density. With every block routed,
the kernel is 3–15 % slower than dense SDPA (0.85×–0.97× between 512 and
8192 tokens). At 50 % density it gains 1.5–1.8× from 2048 tokens up, and at
512 tokens it loses at every density measured. On this chip, dense
attention should run below about 1024 tokens; ltx-2-mlx's `MIN_TOKENS =
4096` is conservative.

## Gate 2: second consumer

ERNIE-Image SFT (MLX port, int8) at 2048² (16,384 image tokens plus the
prompt, 32 heads, head dim 128, bfloat16), one prompt, 2026-11-02. The
harness replaces the port's self-attention by `centroid_threshold_block_mask`
(`τ = 1.0`, `band = 1`) and the kernel, per CFG branch on its unpadded
length, from the third step on.

| Step | Mode | Seconds | Mean density |
|---|---|---|---|
| 0, 1 | dense | 305.0, 304.8 | 1.00 |
| 2, 3 | sparse | 214.4, 217.4 | 0.21 |

A full step (36 layers, CFG batch of two) drops by 29 %, a 1.42× step
speedup. The relative error of the routed attention against dense, on three
layers of the last step, is 0.098, 0.162 and 0.077 (layers 2, 18, 34): this
is the routing error at `τ = 1`, which the kernel reproduces exactly. Whether
it is visible in the image needs a full trajectory against the dense render,
which is the next step for this gate.

## Gate 3: MLX coupling

The kernel inlines `steel/attn/attn.h` and its includes: ten headers
(`attn/{attn,loader,mma,params,transforms}.h`, `defines.h`,
`gemm/params.h`, `utils.h`, `utils/{integral_constant,type_traits}.h`). It
does not include `steel_attention.h` or `steel_attention_nax.h`; it carries
its own copy of the loop. The October catch-up review expected the
rewrites of those two files on MLX main (#4734, #4723) to break the kernel.
They do not touch it: on main at `55626cb` (72 commits after 0.32.4), none
of the ten headers changed, and the kernel built against main's headers
passes the probe and gives identical results.

The CI test the gate asks for is still to write. The approach above is the
template: fetch the headers of the newest allowed `mlx`, build, probe
against dense SDPA.

**Dense backend bug found on the way.** On MLX ≤ 0.32.4, `mx.concatenate`
returns wrong values past 2^31 output elements (fixed upstream in
ml-explore/mlx#4758, unreleased). The dense backend concatenated a
`(B, H, Sq, Sk + Ck)` mask, so at 8 heads × 16,403 tokens it was off by up to
0.14 while the kernel was right. Fixed in mlx-arsenal by chunking the query
rows (PR #122).

## Gate 4: licence

NVlabs/Sana declares Apache-2.0 on its default branch, but the `sol-engine`
branch has no LICENSE file and its routing files carry no SPDX header. The
arsenal router does not depend on that code: the rule is the paper's
(arXiv 2607.24027, Eq. 4 with the diagonal estimator of Appendix B, Eq. 15)
and the compensation is its Eqs. 8–10. The neighbour band and the `1e-6`
floor are implementation choices with no counterpart in the paper. The
kernel itself derives from MLX's MIT steel attention, not from Sana. The
gate is closed on that basis (decision of 2026-11-02).

`sol-engine` also contains an Apple Silicon backend
(`techniques/sparse_backends/sol_attn/mps/`, PyTorch `torch.mps.compile_shader`,
self-contained Metal source, no MLX headers). It would remove the
private-header coupling, but it has the same licence gap, so it is noted
here and not used.
