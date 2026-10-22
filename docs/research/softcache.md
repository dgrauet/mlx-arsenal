# SoftCache adaptive step size

Research notes on SoftCache and the part `mlx-arsenal` exposes. This page
documents a pattern; it does not open an ADR. It extends the
[SeaCache](seacache.md) and [verified feature caching](verified-feature-caching.md)
notes.

## The idea

*Adaptive Step Size Caching for Diffusion Transformers* (Malnick et al.,
arXiv 2610.22369, MIT reference code at
<https://github.com/ShimonMalnick/softcache>, commit 49f00ab).

TeaCache and SeaCache accumulate the relative-L1 change of a cheap probe
(the first-block modulated input, optionally SEA-filtered) and recompute the
transformer once the sum reaches a threshold `tau`. Between computes they
reuse the cached residual. SoftCache changes two things:

- **Unspent budget becomes a longer step.** When the accumulator `A` is well
  under `tau`, the step strides `1 + (J − 1)(1 − A / tau)` scheduled steps
  at once, and the rest of the schedule is re-spaced with fewer nodes. Jumps
  are disabled above `σ = 0.9` and after a sharp probe change.
- **Residual Motion.** Reused steps extrapolate the residual along the
  secant of the last two computed anchors, `r_a + β·λ·(r_a − r_{a−1})`,
  with `λ` the sigma distance from the last anchor relative to the anchor
  spacing.

## What `mlx-arsenal` ships

| API | Role |
|---|---|
| `SoftCacheController(tau, *, max_jump_factor, beta, jump_sigma_max, jump_probe_max)` | decision rule, anchor store, Residual Motion |
| `SoftCacheController.decide(step, sigma, sigma_next, probe)` | returns a `SoftCacheDecision(action, sigma_next, jump_factor)` |
| `SoftCacheController.record(residual, sigma)` / `.residual(sigma)` | store an anchor / extrapolate one |
| `regrid_tail(sigmas, index, target)` | re-space the schedule after a jump |

It is a standalone controller, not a `TeaCacheController` option: jumps
shorten the schedule, and SoftCache does not force a compute at the last
node, which a controller built around a fixed `num_steps` cannot express.
The loop shape is in the `mlx_arsenal.diffusion.softcache` module docstring;
the test suite drives the controller through that loop and compares it with
a transcription of the authors' Flux loop.

## Measured on ERNIE-Image (MLX)

ERNIE-Image SFT (MLX port, int8), 512², CFG 5, the checkpoint's shift-4
schedule (`shift: 4.0` in its scheduler config), two prompts, one seed each.
The reference is Euler at 100 steps. The probe is block 0's modulated input
on the image tokens, `sea_filter`ed (`p = 2`) after step 0; one decision per
step covers both CFG branches; the latent state is kept in float32. SeaCache
is `TeaCacheController(50, thresh, coefficients=None)` on the same probe,
with thresholds chosen to land on the same NFE.

PSNR / SSIM against the reference, first prompt:

| Sampler | NFE | compute / reuse / jump | PSNR | SSIM |
|---|---|---|---|---|
| Euler | 50 | 50 / 0 / 0 | 26.73 | 0.953 |
| SoftCache `tau = 0.3` | 17 | 17 / 7 / 5 | 28.38 | 0.951 |
| SeaCache `0.36` | 17 | 17 / 33 / 0 | 22.31 | 0.918 |
| Euler | 17 | 17 / 0 / 0 | 18.53 | 0.781 |
| Euler on SoftCache's 17 compute sigmas | 17 | 17 / 0 / 0 | 28.53 | 0.946 |
| SoftCache `tau = 0.6` | 11 | 11 / 13 / 6 | 18.38 | 0.769 |
| SeaCache `0.61` | 11 | 11 / 39 / 0 | 18.66 | 0.814 |
| Euler | 11 | 11 / 0 / 0 | 14.69 | 0.694 |
| Euler on SoftCache's 11 compute sigmas | 11 | 11 / 0 / 0 | 18.85 | 0.794 |

The second prompt changes composition between 50 and 100 steps (Euler-50
scores 12.1 dB against the reference), so it was scored against Euler-50:
SoftCache 20.65 dB, SeaCache 14.24 dB and Euler 9.89 dB at 17 NFE; 12.60,
15.55 and 12.17 dB at 11 NFE.

What this shows:

- **It runs cleanly.** No NaN, sharp images, 2.9× and 4.5× fewer transformer
  calls than 50 steps. Every run reaches `σ = 0` through a jump or a reuse,
  with no visible artifact.
- **The gain is a scheduling effect.** Plain Euler on exactly the sigmas
  SoftCache computed matches it (28.53 vs 28.38 dB, 18.85 vs 18.38 dB). The
  reused and extrapolated nodes add essentially nothing. SoftCache keeps the
  dense high-σ start of the 50-step grid; shift-4 Euler with 11–17 steps
  takes coarse early steps and changes the composition.
- **The decisions ignore the prompt.** The action strings are identical for
  both prompts: on ERNIE the probe change is dominated by the timestep
  modulation, so the controller behaves like a fixed schedule.
- **Against SeaCache:** better at `tau = 0.3`, neutral to slightly worse at
  `tau = 0.6`.

So on ERNIE-Image, a fixed schedule that is dense early and coarse late
gets the same result without a controller. SoftCache earns its place where
the probe actually tracks content; measure that on the target model, with
the "Euler on the computed sigmas" control, before adopting it.

## Deviations and notes

- **Linear regrid.** `regrid_tail` re-spaces the tail linearly in `σ`, as
  the reference does, so a shifted schedule loses its shift after the first
  jump. On ERNIE, re-spacing in the shifted parameterization instead
  changed PSNR by +0.05 to +0.9 dB (SSIM both ways) and saved one NFE at
  `tau = 0.3`: not a defect, but a caller can regrid its own way and only
  use `decide` / `residual`.
- **Precision.** The reference divides the relative-L1 distance in float64;
  `mlx-arsenal` divides in float32. Decisions match the transcription
  exactly on the test models; latents agree within `1e-4`, the difference
  being scaled by the jump stride.
- **CFG.** The benchmark takes one decision per step for the batched CFG
  pass. With one controller per branch, the branches could jump at
  different steps and the caller would have to reconcile two schedules.

## Out of scope, and why

- **The probe tensor and the sigma schedule**: model- and scheduler-specific,
  the caller's port.
- **The sampling loop**: the schedule is mutated by jumps; the caller owns
  it (no `*Runner`, per the doctrine).

## References

- SoftCache: <https://arxiv.org/abs/2610.22369>, code
  <https://github.com/ShimonMalnick/softcache>
- SeaCache: <https://arxiv.org/abs/2602.18993>
- TeaCache: <https://github.com/ali-vilab/TeaCache>
