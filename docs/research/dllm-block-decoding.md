# dLLM block decoding

Research notes on decoding masked diffusion language models (dLLMs) and how
`mlx-arsenal` exposes the reusable part. This page documents a pattern and
its fit with the library; it does not open an ADR.

## The pattern

Masked (absorbing-state) dLLMs start from a canvas of `[MASK]` tokens and
decode by repeating one step: run the model on the whole canvas, propose a
token for every masked position, and **commit** a subset of the proposals.
Block (semi-autoregressive) decoding restricts each step to one block of
positions, left to right, which enables caching of the finished prefix.

Every framework that implements this — Fast-dLLM, dInfer, SGLang, LMDeploy,
the LLaDA and Dream reference code — splits the step the same way:

1. **Statistics** of the logits: proposed token, its probability, entropy.
2. **A commit rule**: rank the masked positions, then commit a prefix of the
   ranking until a stopping rule fires (a fixed count, a confidence
   threshold, a confidence-dependent count, an entropy budget).
3. **A schedule**: per-step quotas and the block boundaries.

The model forward, the KV cache, the attention mask, special tokens and
stopping stay with the model. That split matches this library's doctrine,
so the commit step is exposed as pure functions and the loop stays with the
caller.

## What `mlx-arsenal` ships

`mlx_arsenal.diffusion`:

| Function | Role | Reference |
|---|---|---|
| `token_stats` | `(x0, prob, entropy)` per position, float32, NaN-safe; greedy or Gumbel-max sampling with an explicit key; `suppress_ids` | Fast-dLLM / LLaDA `add_gumbel_noise` + `low_confidence` |
| `threshold_transfer` | commit `confidence >= τ` (or `> τ` with `strict=True`), force exactly one if none pass | Fast-dLLM `get_transfer_index`, SGLang LowConfidence, LLaDA2.x `generate` (strict) |
| `topk_transfer` | commit the `k` most confident, `k` per row | LLaDA / Fast-dLLM fixed quota |
| `factor_transfer` | longest prefix with `(i+1)(1 − c_(i)) ≤ f` | Fast-dLLM `get_transfer_index_dynamic` |
| `entropy_bound_transfer` | longest low-entropy prefix with `Σ_{j<i} H_(j) ≤ γ` | EB-Sampler, DiffusionGemma acceptance |
| `edit_transfer` | revise committed tokens when `x0 ≠ token` and `p > τ_edit` | LLaDA2.1/2.2 `editing_threshold`, Nemotron-Labs, SGLang JointThreshold |
| `linear_temperature`, `uniform_canvas`, `renoise`, `StableConfidentStopping` | uniform-noise decoding: temperature schedule, random canvas, re-noising, stable-and-confident stop | Hugging Face `generation_diffusion_gemma.py` |
| `transfer_schedule` | `(B, steps)` linear quotas | Fast-dLLM / LLaDA `get_num_transfer_tokens` |
| `block_ranges` | block boundaries, prompt-relative or absolute-aligned | dInfer `BlockIterator` (`start_block_align`) |

`mlx_arsenal.attention.block_causal_mask` builds the mask block-diffusion
models are trained with: bidirectional inside a block, causal across blocks,
aligned on absolute positions.

## Reference loop

A minimal threshold-based block decoder. `model(x)` is the caller's forward
returning `(B, L, V)` logits; everything model-specific (mask id, attention
mask, cache, EOS handling) is the caller's. The test suite executes this
exact block.

<!-- reference-loop -->
```python
import mlx.core as mx

from mlx_arsenal.diffusion import block_ranges, threshold_transfer, token_stats


def decode(model, prompt, *, gen_len, block_len, mask_id, threshold=0.9):
    """Fill `gen_len` masked tokens after `prompt` (B, P), one block at a time."""
    B, P = prompt.shape
    x = mx.concatenate([prompt, mx.full((B, gen_len), mask_id, dtype=prompt.dtype)], axis=1)
    positions = mx.arange(P + gen_len)
    forwards = 0
    for start, end in block_ranges(P, gen_len, block_len):
        in_block = (positions >= start) & (positions < end)
        while True:
            candidates = (x == mask_id) & in_block
            if not mx.any(candidates).item():
                break
            logits = model(x)  # caller: attention mask, KV cache, ...
            forwards += 1
            stats = token_stats(logits, suppress_ids=[mask_id])
            commit = threshold_transfer(stats.prob, candidates, threshold)
            x = mx.where(commit, stats.x0.astype(x.dtype), x)
    return x, forwards
```

Swap the commit rule to change the sampler:

- LLaDA fixed quota: `topk_transfer(stats.prob, candidates, schedule[:, step])`
  with `schedule = transfer_schedule(mx.sum(candidates, axis=1), steps)`
  computed at the start of each block.
- Fast-dLLM factor: `factor_transfer(stats.prob, candidates, factor)`.
- EB-Sampler: `entropy_bound_transfer(stats.entropy, candidates, gamma)`.
- LLaDA2.x reference: `threshold_transfer(stats.prob, candidates, 0.95, strict=True)`.
- Sampling: `token_stats(logits, temperature=t, key=k)` with a fresh key per
  step (`mx.random.split`).

**Attention layout is model-specific.**

- Block-causal models (LLaDA2.x, SDAR): use `block_ranges(..., align=True)`
  and `block_causal_mask(L, block_len)` over the window. The LLaDA2.x
  reference re-runs the full window each step with exactly this mask; a
  prefix KV cache of the finished blocks is equivalent.
- Nemotron-Labs Diffusion (`causal_context=True`) is **not** block-causal:
  its prefix is encoded causally token by token (`causal_mask`, also used to
  seed each block's first token), and only the current block attends
  bidirectionally on top of that cache.
- Bidirectional models (LLaDA 1.x, Dream) attend to the whole canvas.

**Mask conventions.** `mlx_arsenal.attention` masks are additive (`0`
attend, `-inf` blocked), as `mx.fast.scaled_dot_product_attention` expects.
Runtimes that take a boolean or 0/1 *keep* mask — mlx-vlm's `mask=`,
Hugging Face `attention_mask` — read an additive mask inverted, silently;
pass `block_causal_mask(...) == 0` to them.

## Token editing (LLaDA2.1)

LLaDA2.1/2.2 and Nemotron-Labs Diffusion also revise committed tokens: after
the mask commit, a committed position outside the prompt takes the new
prediction when it differs and its confidence exceeds `editing_threshold`.
A block ends when it has no mask and no edit happened, or after
`max_post_steps` extra passes once it is full. This is the LLaDA2.1
reference loop minus its `eos_early_stop` option and final cut at the first
EOS (it re-runs the window each step with `block_causal_mask`, which is the
model's job here). Like the reference, it is written for batch 1: with
`B > 1` the rows share the loop's termination, so a finished row keeps
receiving edit passes while another row still has masks. The test suite
executes this exact block.

<!-- editing-loop -->
```python
import mlx.core as mx

from mlx_arsenal.diffusion import block_ranges, edit_transfer, threshold_transfer, token_stats


def decode_with_editing(model, prompt, *, gen_len, block_len, mask_id, threshold=0.95,
                        editing_threshold=0.9, max_post_steps=16):
    """LLaDA2.1-style block decoding with token-to-token editing."""
    B, P = prompt.shape
    total = -(-(P + gen_len) // block_len) * block_len  # whole blocks
    x = mx.concatenate([prompt, mx.full((B, total - P), mask_id, dtype=prompt.dtype)], axis=1)
    positions = mx.arange(total)
    forwards = 0
    for start, end in block_ranges(P, total - P, block_len, align=True):
        in_block = ((positions >= start) & (positions < end) & (positions >= P))[:end]
        post_steps = 0
        while True:
            window = x[:, :end]
            masked = (window == mask_id) & in_block
            has_mask = mx.any(masked).item()
            if not has_mask:
                post_steps += 1
                if post_steps > max_post_steps:
                    break
            stats = token_stats(model(window))  # caller: block_causal_mask(end, block_len)
            forwards += 1
            commit = threshold_transfer(stats.prob, masked, threshold, strict=True)
            editable = in_block & ~masked
            edit = edit_transfer(stats.x0, stats.prob, window, editable, editing_threshold)
            window = mx.where(commit | edit, stats.x0.astype(x.dtype), window)
            x = mx.concatenate([window, x[:, end:]], axis=1)
            if not has_mask and not mx.any(edit).item():
                break
    return x[:, P : P + gen_len], forwards
```

## Uniform-noise decoding (DiffusionGemma)

DiffusionGemma does not use a mask token. A canvas starts as uniformly
random tokens; at every step the model predicts every position, the
lowest-entropy predictions are accepted under an entropy budget
(`entropy_bound_transfer` over all positions — nothing is frozen, accepted
tokens can change next step), and every other position is re-noised. The
temperature decays linearly with the number of remaining steps and entropy
is measured on the temperature-scaled logits; the output is the last argmax
canvas. The test suite executes this exact block.

<!-- renoise-loop -->
```python
import mlx.core as mx

from mlx_arsenal.diffusion import (
    StableConfidentStopping,
    entropy_bound_transfer,
    linear_temperature,
    renoise,
    token_stats,
    uniform_canvas,
)


def decode_uniform(model, *, batch, canvas_len, vocab_size, num_steps=48, entropy_bound=0.1,
                   t_min=0.4, t_max=0.8, stability_threshold=1, confidence_threshold=0.005):
    """Decode one canvas: accept low-entropy tokens, re-noise the rest."""
    canvas = uniform_canvas((batch, canvas_len), vocab_size)
    stop = StableConfidentStopping(stability_threshold, confidence_threshold)
    everywhere = mx.ones((batch, canvas_len), dtype=mx.bool_)
    steps = 0
    for remaining in range(num_steps, 0, -1):
        steps += 1
        z = model(canvas) / linear_temperature(remaining, num_steps, t_min=t_min, t_max=t_max)
        stats = token_stats(z)  # argmax canvas and entropy of softmax(z)
        sample = mx.random.categorical(z).astype(mx.int32)
        accepted = entropy_bound_transfer(stats.entropy, everywhere, entropy_bound)
        canvas = renoise(sample, accepted, uniform_canvas(canvas.shape, vocab_size))
        if mx.all(stop(stats.x0, stats.entropy)).item():
            break
    return stats.x0, steps
```

The model call hides DiffusionGemma's self-conditioning (the previous
step's scaled logits are fed back) and its causal encoding of finished
canvases. Draws use the global PRNG in the reference's order (canvas, then
per step the sample and the noise), so a loop seeded like a reference
reproduces its draws. For simplicity the whole batch stops together; the
reference freezes finished rows individually.

## Model coverage

Covered: DiffusionGemma (uniform-noise decoding; self-conditioning and the
encoder cache stay model-side), LLaDA 1.x/1.5, Dream (the caller applies Dream's logit shift),
LLaDA2.0/2.1 (with token editing), SDAR/TraDo, Nemotron-Labs Diffusion
(diffusion mode, with token editing), LLaDA-UI, LLaDA2.2 with substitution
editing but without its DELETE/INSERT edits.

Not covered by v1:

- **LLaDA2.2 DELETE/INSERT** edits change the canvas length; only
  substitution editing (`edit_transfer`) is covered.

## Deviations from the references

- **Confidence in float32.** References disagree: Fast-dLLM uses float64
  (unavailable on the MLX GPU), LLaDA2.x casts logits to float32 (as here),
  and Nemotron-Labs Diffusion softmaxes bfloat16 logits, which creates exact
  confidence ties that `torch.topk` breaks arbitrarily. float32 keeps the
  real ordering; near-ties can resolve differently from a given reference.
- **Ties:** sorting is stable, so equal scores commit the lower position
  first; the forced token of `threshold_transfer` is the first maximal one.
  dInfer forces with `max − 1e-5`, which can commit several near-tied
  tokens; Fast-dLLM and SGLang force exactly one, as here.
- **`factor_transfer`** does not reproduce the reference's off-by-one, which
  commits every candidate when only the last sorted one fails.
- **`suppress_ids`** are removed before the softmax, so `prob` and `entropy`
  are over the remaining vocabulary (dInfer `rm_mask`).
- **Per-row quotas.** Dream averages the per-step quota over the batch;
  every function here works per row.

## Validation on real models (2026-09)

A throwaway bench on an M2 Pro (32 GB) challenged the primitives before
release, with greedy decoding, 3 prompts, 64 generated tokens, and both the
threshold (0.9) and the top-k schedule samplers.

- **End to end against the reference, float32.** Nemotron-Labs-Diffusion-3B
  was run with NVIDIA's PyTorch `generate` (MPS) and with its algorithm
  rebuilt from these primitives on mlx-vlm's MLX forward. The two forwards
  agree to 9e-5 in logits. All 6 runs produce **identical tokens** with
  identical forward counts (up to the EOS where the reference stops).
- **Same forward as mlx-vlm, bfloat16.** Aligning only the caller-side
  plumbing with mlx-vlm's `generate`, the primitives produce identical
  tokens on all 9 runs:
  - Nemotron-3B, with mlx-vlm's bfloat16 confidence passed in as the score;
  - LLaDA2.1-mini 4-bit, with the same prefix cache and `strict=True`.
- **Decision parity on the reference's own bfloat16 logits.** At every
  step of NVIDIA's `generate`, the same logits were fed to `token_stats` and
  the matching rule. Over 227 steps, proposed tokens agree 100 % and
  commits agree on 212. The other 15 are exact ties in the reference's
  bfloat16 confidence, which `torch.topk` breaks arbitrarily.

End-to-end runs across frameworks in bfloat16 did diverge on 1 of 3
prompts for each model. In both cases the cause was forward numerics
flipping a near-tie (0.327 vs 0.326), not the commit rules:

- For Nemotron, the MLX and PyTorch bfloat16 forwards differ by about 0.01
  in probability at that step; in float32 the run matches.
- For LLaDA2.1-mini, the 4-bit MoE forward gives different logits for the
  same tokens depending on how many rows are processed at once. An
  uncached window and a prefix cache therefore disagree; with the same
  cache path the run matches.

## Out of scope, and why

- **KV caches** (prefix / dual cache, refresh policies): a training-free
  cache is exact only for block-causal-trained models, and approximate on
  bidirectional ones. That is a model-family property, and cache objects
  would chase each model's plumbing. `block_causal_mask` is the part that
  generalizes.
- **Attention-guided rankings and caches** (ADAS, Flash-dLLM): they need
  attention probabilities, which `mx.fast.scaled_dot_product_attention`
  does not return.
- **Learned routers** (EB-Decode) and model retraining: per-model
  calibration.
- **Adaptive block length** (PACE-dLLM) and verification of inserted spans
  (CARVE): logit-only and cheap, but without public code or independent
  replication yet — watch-list.
- **A `generate()` wrapper**: every model has its own loop (see above).

## References

- Fast-dLLM: <https://github.com/NVlabs/Fast-dLLM> (`v1/llada/generate.py`)
- LLaDA: <https://github.com/ML-GSAI/LLaDA>
- dInfer: <https://github.com/inclusionAI/dInfer>
- SGLang dLLM: <https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/dllm>
- EB-Sampler: <https://arxiv.org/abs/2505.24857>
- mlx-vlm diffusion engine: <https://github.com/Blaizzy/mlx-vlm>
