# Posterior-mean-capped guidance (PMC-CFG)

Research notes on PMC-CFG and the function `mlx-arsenal` exposes. This page
documents a pattern; it does not open an ADR.

## The idea

*Classifier-Free Guidance in Flow Matching: Non-Autonomous Potentials,
Overshoot, and Posterior-Mean Control* (arXiv 2609.24287, September 2026,
no public code).

Classifier-free guidance extrapolates `v_u + λ(v_c − v_u)`. For a flow-matching
model each velocity implies a clean-sample estimate, the posterior mean
`m = x + (1 − t)·u` (paper convention). At large `λ` the guided estimate
overshoots, which shows up as over-saturated, over-contrasted images. PMC-CFG
keeps the guidance direction but, per sample and per step, takes the largest
increment `β ∈ [0, λ − 1]` for which the guided posterior mean stays within
`Γ` times the norm of the conditional one:

```text
β* = max{β ∈ [0, λ − 1] : ‖m_c + β·Δ‖ ≤ Γ·‖m_c‖},   Δ = m_c − m_u
v  = v_c + β*·(v_c − v_u)
```

The constraint is a quadratic in `β` with a closed-form root. The paper calls
the cap *self-releasing*: as sampling ends, the conditional and unconditional
estimates agree (`Δ → 0`) and the nominal scale comes back without a
schedule. No extra model evaluations.

Reported: `Γ = 1.10` on a GMM toy, `Γ = 1.05`–`1.10` on SiT-XL/2 ImageNet-256
(FID 6.22 → 5.07 at `λ = 2.5`), and on SD3.5 Medium at 10 steps it prevents
the saturation spike at `λ = 9`.

## What `mlx-arsenal` ships

`mlx_arsenal.diffusion.posterior_mean_capped_guidance(cond, uncond, scale, *,
x, sigma, cap)`, next to `classifier_free_guidance`, with the same `scale`
meaning (`uncond + scale·(cond − uncond)` when the cap does not bind).

It uses the diffusers convention: `σ = 1` is noise, `v = noise − x0`, and the
posterior mean is `m = x − σ·v` (the `x0` estimate). Norms are per sample
over all non-batch axes, in float32; `sigma` is a float, a 0-d array (e.g.
`scheduler.sigmas[i]`) or a `(B,)` array.

A caller in the paper's convention (`t = 1` is data, `u = x0 − noise`) passes
`-u` velocities and `σ = 1 − t`, and negates the result.

## Measured on ERNIE-Image (MLX)

ERNIE-Image SFT (MLX port, int8), 512², its default shift-3 schedule at 28
steps, two prompts, one seed each. Saturation is the mean HSV saturation,
"clipped" the share of pixels with a channel at 0 or 255.

| Guidance | HSV sat. p1 | clipped p1 | HSV sat. p2 | clipped p2 |
|---|---|---|---|---|
| CFG `λ = 5` | 0.174 | 2.9 % | 0.515 | 0.6 % |
| CFG `λ = 9` | 0.172 | 6.9 % | 0.502 | 1.5 % |
| PMC `λ = 9`, `Γ = 1.05` | 0.169 | 0.4 % | 0.515 | 4.3 % |
| PMC `λ = 9`, `Γ = 1.10` | 0.174 | 0.5 % | 0.496 | 3.4 % |
| PMC `λ = 5`, `Γ = 1.05` | 0.165 | 0.3 % | 0.508 | 3.3 % |

The mechanism behaves as described: `β` starts near 0.06 and releases by
itself to the nominal `λ − 1` over the last five steps. But ERNIE-Image does
not over-saturate at `λ = 9`, so there is no spike to remove, and the clipped
share goes down on one prompt and up on the other. Because early steps are
barely guided, PMC-CFG also changes the composition (layout, camera, light):
on the first prompt the requested "dawn, soft light" mostly disappears.
Evaluate it per model, with a prompt-adherence score, before adopting it.

## Deviations and notes

- **`Γ ≥ 1` only.** Appendix D.4 has a branch for a negative discriminant,
  which only `Γ < 1` can reach; there even the unguided `m_c` breaks the cap
  and the constraint set can be empty. The paper uses `Γ = 1.05`–`1.10`, so
  the function requires `Γ ≥ 1`; then the discriminant is non-negative and
  the root lies in `[0, ∞)` (Cauchy-Schwarz).
- **`a = 0`** (no gap, e.g. `σ = 0`): nominal guidance, as the paper.

## Out of scope, and why

- **Choosing `Γ` per model**: a hyperparameter, like `λ`.
- **CFG batching and the model call**: the caller's pipeline.

## References

- PMC-CFG: <https://arxiv.org/abs/2609.24287>
- CFG-Zero* (a related flow-matching guidance fix):
  <https://arxiv.org/abs/2503.18886>
