# ADR-0002 gate log

Status of the four gates that [ADR-0002](../adr/0002-block-sparse-attention-kernel.md)
sets before the ltx-2-mlx Sol-Attn Metal kernel can move behind
`block_sparse_attention`. Each entry is dated; the kernel stays in the port
until all four pass.

## Summary (2026-11-06)

| Gate | Status |
|---|---|
| 1. Hardware and shapes | Partial: M1 Max and M2 Pro pass at head dim 64 and 128, float16 and bfloat16. No chip with GPU neural accelerators (M5 family) is available. |
| 2. Second consumer | Passed: ERNIE-Image at 2048², full 20-step render, 1.34× end to end against dense. The image is clean but not the dense image (PSNR 23.7 dB); the text-token block must stay routed (see below). |
| 3. MLX coupling | Passed: the kernel is staged in mlx-arsenal (private, not dispatched); CI builds it against the newest allowed `mlx` on every run and against MLX main every week. |
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
is the routing error at `τ = 1`, which the kernel reproduces exactly.

**Full render (2026-11-04).** 20 steps at 2048², seed 0, guidance 5, the
first two steps dense and the other 18 routed. The router runs on the
arsenal build with the fix of PR #122.

| Run | Total seconds | Speedup | PSNR vs dense | PSNR at 1/32 |
|---|---|---|---|---|
| dense | 6111 | 1.00× | | |
| routed, `τ = 1` | 4528 | 1.35× | 22.4 dB | 25.0 dB |
| routed, `τ = 1`, text block always kept | 4577 | 1.34× | 23.7 dB | 26.8 dB |

Routed steps take 214–224 s at a mean density of 0.21. The total includes
the text encoder and the VAE decode. On the last step the kernel and the
arsenal dense backend agree to four digits in relative error against dense
attention (0.063, 0.054 and 0.069 on layers 2, 18 and 34 for the last run).

Neither routed image shows block artefacts or blur. Both keep the dense
image's layout and light, which the two dense steps fix, but the fox moves
to the right and turns to face the camera. Low-pass PSNR stays near 25 dB
because the subject itself differs, not because of fine-detail noise. At
`τ = 1` the routed trajectory is a different sample of the same prompt, not
an approximation of the dense one; a caller who needs the dense image needs
a lower `τ` or more dense steps.

**Text-token block.** The first routed render had a dark band 3–5 px wide
on the left edge, up to 47 grey levels darker in the lower half, absent from
the dense render and from the other three edges. A per-token comparison of
routed and dense attention at step 2, over all 36 layers, located it:

- ERNIE places the text tokens after the image tokens and gives them the
  RoPE position `(y, x) = (0, 0)`. Image tokens in column 0 share `x = 0`
  and give the text tokens more attention than other columns (33 % of
  their attention against 15–16 % for columns 1, 64 and 127 on layer 33).
- The 19 text tokens fill one key block of their own. The router decides
  per query block of 64 tokens, of which one is in column 0, and on deep
  layers it drops the text block for about 40 % of the query blocks. Column
  0 then sees the text only through the block's centroid.
- The column-0 error is on average 2.1× the image mean, and 10.5× on layer
  33 of the conditioned branch. Keeping the text block for every query
  block brings it to 1.14× (1.25× on layer 33). It also lowers the mean
  error over the image from 0.123 to 0.116, for 0.004 more density.

With the text block always kept, the band is gone: the left-edge columns
are within 3.4 grey levels of their neighbours, as in the dense render. The
fix belongs to the caller, which takes the union of the routed mask and the
key blocks that hold conditioning tokens. Any model whose conditioning
tokens share RoPE coordinates with image tokens should do the same.

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

**CI tripwire (2026-11-06).** The kernel is staged as
`mlx_arsenal/attention/_sol_kernel.py`: private, and not called by
`block_sparse_attention` (`block_sparse_kernel_available` stays `False`
until gate 1 closes). `tests/test_attention_sol_kernel.py` builds it and
checks it:

- every block routed against `mx.fast.scaled_dot_product_attention`, and
  with skipped blocks against the dense backend of `block_sparse_attention`,
  in float16, bfloat16 and float32 at head dim 64 and 128, with a partial
  last block, per-batch and per-head routes, and more than one centroid
  tile;
- the set of inlined headers against the ten above, so that a change in
  that set fails even when the kernel still builds.

The CI workflow installs the newest `mlx` that `pyproject.toml` allows, so
an MLX release that breaks the kernel fails the next run, including the
Dependabot bump. The weekly `mlx-main-kernel` workflow runs the same file
against the steel headers of MLX main (`MLX_ARSENAL_STEEL_INCLUDE`), so a
breaking change shows up before the release that ships it. Locally both
pass with MLX 0.32.4 and with main at `c69303b`. With every block routed at
head dim 128 the kernel matches dense SDPA bit for bit (same steel loop);
the worst errors otherwise are 2.4e-4 (float16), 2.0e-3 (bfloat16) and
1.0e-7 (float32).

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
