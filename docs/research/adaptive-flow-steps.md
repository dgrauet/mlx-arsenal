# Adaptive flow-matching steps (CAT-Flow)

Research notes on CAT-Flow and the step controller `mlx-arsenal` exposes.
This page documents a pattern; it does not open an ADR.

## The idea

*CAT-Flow: Curvature-Adaptive sTeps for Flow Matching* (arXiv 2609.01746,
September 2026, CC BY 4.0, no public code).

Flow-matching samplers integrate `dx/dt = u(x, t)` from noise (`t = 0`) to
data (`t = 1`) on a fixed schedule. CAT-Flow instead picks every Euler step
from the velocity the model just returned — no extra evaluations:

- **CAT-OT** — the trajectory's acceleration, by finite difference:
  `dt = λ / ‖(u_k − u_{k−1}) / dt_{k−1}‖₂`; the first step is `Δ_min`.
- **CAT-OV** — Adam-like moments of the scaled velocity `(1 − t)·u`:
  `m1 ← β m1 + (1 − β)(1 − t)u`, `m2 ← β m2 + (1 − β)((1 − t)u)²`,
  `dt = λ / sqrt(‖m2 − m1²‖₂)`, with a `sqrt(1 − β)` bias correction on the
  first step.

Each step is clipped to `[Δ_min, 1 − t]`. The paper reports comparable
quality with up to 40 % fewer steps than diffusers' dynamic-shift schedule
on FLUX.1-dev, FLUX.1-Krea-dev, FLUX.1-schnell and SD3.5-large, with
`λ = 1.5`–`2`, `β = 0.3`, `Δ_min = 0.01`, and 2–3 small warm-up steps
(early velocities are unreliable).

## What `mlx-arsenal` ships

`mlx_arsenal.diffusion.CurvatureAdaptiveStepper(scale, *, mode="ov",
beta=0.3, dt_min=0.01, dt_max=None, warmup_steps=0, t_start=0.0)`:
`step(velocity) -> dt` once per step, `t`, `steps`, `done`, `reset()`.

Both rules only look at norms of velocity differences and squares, so they
are invariant to the velocity's sign: a diffusers-convention output (`σ =
1 − t`, `x ← x + (σ_next − σ)·v`, `v = −u`) goes in unchanged.

## Recipe

The test suite executes this block on a Gaussian flow whose velocity and
exact solution are known in closed form.

<!-- adaptive-steps-recipe -->
```python
from mlx_arsenal.diffusion import CurvatureAdaptiveStepper


def adaptive_euler(velocity_fn, x, stepper):
    """Euler flow-matching sampling with CAT-Flow step sizes (diffusers sigma convention)."""
    stepper.reset()
    nfe = 0
    while not stepper.done:
        sigma = 1.0 - stepper.t  # paper time t = 1 - sigma
        v = velocity_fn(x, sigma)  # model output, CFG already applied
        nfe += 1
        dt = stepper.step(v)
        x = x - dt * v  # sigma decreases by dt
    return x, nfe
```

Models conditioned on a timestep in `[0, 1000]` get `1000 · sigma`. The
step count is not known in advance: size any per-step buffers dynamically.

## Deviations and notes

- **`λ` depends on the latent size.** `‖·‖₂` is taken over the whole latent,
  so its value grows with resolution and channel count; the paper's
  `1.5`–`2` are for FLUX / SD3.5 latents. Re-tune `scale` per model family
  and resolution — it is the sampler's only knob, not a calibration run.
- **Batches.** The paper has one sample. Here norms are per sample and the
  smallest step wins, so a batch shares one timeline and batch 1 matches the
  paper exactly.
- **`dt_max`.** Not in the paper: an optional cap. Without it a flat
  stretch (zero norm) jumps straight to `t = 1`, as the paper's clip does.
- **Warm-up.** The paper describes "a few small fixed steps" outside its
  algorithms; `warmup_steps` forces `Δ_min` on the first steps while the
  state keeps updating (as the algorithms' loop would).
- **Euler only.** The paper evaluates Euler; nothing here assumes another
  solver.

## Out of scope, and why

- **The sampling loop, CFG and the model call**: the caller's pipeline (the
  `*Runner` pattern the doctrine excludes).
- **Per-model `λ` tuning**: a hyperparameter the caller picks.

## References

- CAT-Flow: <https://arxiv.org/abs/2609.01746>
- Dynamic shifting baseline (Esser et al., 2024): <https://arxiv.org/abs/2403.03206>
