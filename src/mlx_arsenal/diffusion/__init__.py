"""Diffusion primitives: timestep embeddings, schedulers, samplers, caching."""

from .attention_cache import PerHeadAttentionCache, PerLayerAttentionCache, splice_heads
from .cfg_skip import (
    CFGSimilarityProfiler,
    CFGSkipController,
    cfg_head_similarity,
    cfg_skip_mask,
)
from .ddim import DDIMScheduler
from .mask_reuse import HeadMaskCache, pooled_qk, qk_drift
from .masked_decode import (
    TokenStats,
    block_ranges,
    edit_transfer,
    entropy_bound_transfer,
    factor_transfer,
    threshold_transfer,
    token_stats,
    topk_transfer,
    transfer_schedule,
)
from .samplers import classifier_free_guidance, euler_step
from .schedulers import (
    FlowMatchEulerDiscreteScheduler,
    dynamic_shift_schedule,
    get_sampling_sigmas,
)
from .seacache import sea_filter
from .teacache import TeaCacheController
from .timestep import TimestepEmbedding, get_timestep_embedding
from .uniform_decode import (
    StableConfidentStopping,
    linear_temperature,
    renoise,
    uniform_canvas,
)
from .verified_cache import VerifiedFeatureCache, geometric_threshold
from .window_residual import WindowResidualController

__all__ = [
    "CFGSimilarityProfiler",
    "CFGSkipController",
    "DDIMScheduler",
    "FlowMatchEulerDiscreteScheduler",
    "HeadMaskCache",
    "PerHeadAttentionCache",
    "PerLayerAttentionCache",
    "StableConfidentStopping",
    "TeaCacheController",
    "sea_filter",
    "TimestepEmbedding",
    "TokenStats",
    "VerifiedFeatureCache",
    "WindowResidualController",
    "block_ranges",
    "cfg_head_similarity",
    "cfg_skip_mask",
    "classifier_free_guidance",
    "dynamic_shift_schedule",
    "edit_transfer",
    "entropy_bound_transfer",
    "euler_step",
    "factor_transfer",
    "geometric_threshold",
    "get_sampling_sigmas",
    "get_timestep_embedding",
    "linear_temperature",
    "pooled_qk",
    "qk_drift",
    "renoise",
    "splice_heads",
    "threshold_transfer",
    "token_stats",
    "topk_transfer",
    "transfer_schedule",
    "uniform_canvas",
]
