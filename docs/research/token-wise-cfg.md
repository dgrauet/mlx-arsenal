# Token-wise guidance scales (TrajCFG)

Research notes on TrajCFG and the function `mlx-arsenal` exposes. This page
documents a pattern; it does not open an ADR.

## The idea

*TrajCFG: Token-Wise Classifier-Free Guidance via Denoising Trajectory
Feedback* (arXiv 2610.14153, October 2026).

Classifier-free guidance applies one scale `s` to every token. TrajCFG reads
how each token's guided velocity has been turning over the last steps and
gives it its own scale. A token whose velocity keeps its direction is settled
and gets a low scale; one whose velocity swings gets a high one. The scale
comes from the history of guided velocities the solver actually consumed,
so it costs no extra model evaluation.

Per token (a vector along the channel axis), with `v̂_k` the guided velocity
at step `k` and `τ_k` its noise level:

```text
cos_δ(a, b) = clip(⟨a, b⟩ / (max(‖a‖, δ)·max(‖b‖, δ)), −1, 1)
g(q)        = s_min + (s_max − s_min)·clip((1 − q) / (1 − ρ_lo), 0, 1)

k = 0, 1:  s = s_base
k = 2:     s = g(cos_δ(v̂_{k−2}, v̂_{k−1}))
k ≥ 3:     s = 0.8·g(q_short) + 0.2·g(q̃_long)
           q_short = cos_δ(v̂_{k−2}, v̂_{k−1})
           q̃_long  = cos((Δ_short / Δ_long)·arccos(cos_δ(v̂_{k−3}, v̂_{k−1})))
           Δ_short = |τ_{k−1} − τ_{k−2}|,  Δ_long = |τ_{k−1} − τ_{k−3}|

v̂_k = v_u + s ⊙ (v_c − v_u)
```

The long-range term rescales the two-step angle to a one-step interval, so a
non-uniform schedule does not read as extra curvature. The paper sets
`ρ_lo = 0.95` and, per model, `(s_base, s_min, s_max)`: `(5, 4, 9)` for
Qwen-Image at 30 steps and `(7, 6, 9)` for SD3 at 28. The reported
text-to-image gains are small: GenEval +0.01, DPG-Bench +0.08 to +0.43.

## What `mlx-arsenal` ships

`mlx_arsenal.diffusion.trajectory_guidance_scale(history, sigmas, *,
min_scale, max_scale, rho_lo=0.95, long_weight=0.2, eps=1e-6)` returns the
`(..., 1)` float32 scale map for the next step. `history` holds the last two or
three guided velocities, oldest first, and `sigmas` the matching noise levels. With
two entries only the short-range term is used (the `k = 2` case).
`classifier_free_guidance` now accepts an array `scale`, so the caller
writes:

```python
hist, hsig = [], []
for i, sigma in enumerate(sigmas):
    v_u, v_c = model(...)  # channels-last
    if len(hist) >= 2:
        s = trajectory_guidance_scale(hist, hsig, min_scale=4.0, max_scale=9.0)
    else:
        s = 5.0
    v = classifier_free_guidance(v_c, v_u, s)
    hist, hsig = [*hist, v][-3:], [*hsig, sigma][-3:]
    x = solver_step(x, v, ...)
```

The loop, the history buffer and the set of adaptive steps stay with the
caller, as with the other guidance helpers.

## Measured on ERNIE-Image (MLX)

ERNIE-Image SFT (MLX port, int8), 512², 50 steps, linear σ with shift 4,
seed 0, sixteen GenEval-style prompts (colour binding, counting, position,
text, material, scene, portrait). Plain CFG at `s = 5` against TrajCFG at
`s_base = 5, s_min = 4, s_max = 9, ρ_lo = 0.95` (the paper's Qwen-Image set,
unchanged), at matched NFE (100 per image). Each prompt has three to five
yes/no questions (59 in total), asked to Gemma 3 12B (4-bit) and then checked
by eye. "Sample-wise" replaces each TrajCFG map by its mean over tokens
(the paper's Q1 ablation) and otherwise runs the same loop.

| Guidance | judge yes | after visual check | s / image |
|---|---|---|---|
| CFG `s = 5` | 56 / 59 | 56 / 59 | 802 |
| TrajCFG | 54 / 59 | 57 / 59 | 759 |
| Sample-wise TrajCFG | 55 / 59 | 57 / 59 | 797 |

The three questions the judge failed only for TrajCFG (a yellow strawberry,
a cat left of a plant, a sign reading "OPEN") are judge errors: both images
satisfy them. The only real difference is a count: on "three green apples"
CFG draws two and TrajCFG draws three. Every mode misses "four birds"
(three) and "five cups" (four). The two images of each pair keep the same
composition, colours and subject; there is no over-saturation and no
artefact. The time difference is noise (shared machine); the scale map is
a handful of element-wise ops per step.

The scale is not degenerate. Averaged over the sixteen prompts, the mean
scale is about 8.7 at step 6 (61 % of tokens at `s_max`). It drops to 4.8 over
steps 26–34 (p5 4.2, p95 about 7), then rises again to 7.9 at the last step.
The per-token spread is real: at step 14 the p5–p95 range is 4.5–8.7. On the
first prompt the map is high on the out-of-focus background and low on the
objects and the table. The mean scale over the run is 6.1, above CFG's 5,
so the one count fix may come from more guidance overall rather than from
the token-wise split. The paper's own token-wise check (its Q1 ablation,
against a sample-wise variant that averages each map over tokens) is on
video only: +0.27 to +0.54 VBench Total on Wan 2.1 and 2.2.

On ERNIE-Image the sample-wise control settles it: it also draws three
apples, fails the same two counts, and on fifteen of sixteen prompts its
image is closer to TrajCFG's than to CFG's (mean absolute pixel difference
2–16 against TrajCFG on thirteen prompts; 4–65 between TrajCFG and CFG).
Whatever TrajCFG changes here comes from its per-step mean scale, high
early and late and low in the middle. The token-wise split adds nothing
measurable on this model and bench.

The verdict is neutral to slightly positive, the same bar as
[PMC-CFG](posterior-mean-cfg.md). The paper's set transfers to ERNIE-Image
without tuning and without regression, but sixteen prompts and one seed
cannot resolve a gain of the size the paper reports. Evaluate it per model;
a caller who only wants the schedule can average the map over tokens.

## Deviations and notes

- **`δ`** is not given in the paper; `eps = 1e-6` floors each norm, so a
  zero velocity gives cosine 0 and therefore `s_max` rather than a NaN.
- **The adaptive step set `𝒜`** (which steps use the map) is left to the
  caller. The paper's Algorithm 1 adapts every step from `k = 2` on.
- **Token axis.** A token is a vector along the last axis, so the function
  works on channels-last latents, packed sequences or video tokens alike. The
  paper uses whatever latent sites the backbone exposes and does not compare
  choices.
- **Clean-room.** The reference code has no licence; the function is written
  from the paper's equations and Algorithm 1, and the tests check it against
  a float64 NumPy transcription of that algorithm.

## Out of scope, and why

- **Per-model `(s_base, s_min, s_max)`**: hyperparameters, like `s` in CFG.
- **The history buffer and the model call**: the caller's loop.

## References

- TrajCFG: <https://arxiv.org/abs/2610.14153>
- Project page: <https://jyliu-98.github.io/trajcfg-project-page/>
