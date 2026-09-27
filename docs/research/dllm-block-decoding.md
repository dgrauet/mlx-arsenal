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
| `threshold_transfer` | commit `confidence >= τ`, force exactly one if none pass | Fast-dLLM `get_transfer_index`, SGLang LowConfidence |
| `topk_transfer` | commit the `k` most confident, `k` per row | LLaDA / Fast-dLLM fixed quota |
| `factor_transfer` | longest prefix with `(i+1)(1 − c_(i)) ≤ f` | Fast-dLLM `get_transfer_index_dynamic` |
| `entropy_bound_transfer` | longest low-entropy prefix with `Σ_{j<i} H_(j) ≤ γ` | EB-Sampler, DiffusionGemma acceptance |
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
- Sampling: `token_stats(logits, temperature=t, key=k)` with a fresh key per
  step (`mx.random.split`).

For block-causal models (LLaDA2.x, SDAR, Nemotron-Labs Diffusion) use
`block_ranges(..., align=True)` and pass
`block_causal_mask(L, block_len)` to the model; the finished prefix can
then be cached exactly.

## Model coverage

Covered: LLaDA 1.x/1.5, Dream (the caller applies Dream's logit shift),
LLaDA2.0/2.1, SDAR/TraDo, Nemotron-Labs Diffusion (diffusion mode),
LLaDA-UI, LLaDA2.2 without its DELETE/INSERT edits.

Not covered by v1:

- **DiffusionGemma** decodes differently: uncommitted tokens are re-noised
  (uniform noise, not a mask token) under a temperature schedule, with
  self-conditioning. `entropy_bound_transfer` is its acceptance rule, but
  the renoise step and schedule are not shipped yet.
- **Token editing** (LLaDA2.1/2.2 `editing_threshold`, Nemotron-Labs,
  SGLang JointThreshold) re-opens committed tokens; LLaDA2.2's
  DELETE/INSERT also changes the canvas length.

## Deviations from the references

- **float32, not float64.** Fast-dLLM computes Gumbel noise and softmax in
  float64; the MLX GPU has no float64. Near-ties can resolve differently.
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
