"""DDIM (Denoising Diffusion Implicit Models) scheduler — MLX port.

Matches the diffusers ``DDIMScheduler`` / ``CogVideoXDDIMScheduler``
behaviour for the deterministic case (``eta=0``). Supports the two
common prediction types (``epsilon`` and ``v_prediction``), the two
common spacing strategies (``leading``, ``trailing``), and optional
zero-SNR rescaling of the beta schedule.

Use this for ports of DDPM-trained models (CogVideoX, Stable Diffusion
1.x / 2.x, ...). For flow-matching models (LTX, Hunyuan-DiT,
ERNIE-Image), see :class:`FlowMatchEulerDiscreteScheduler` instead.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

from .._typing import item_int

__all__ = ["DDIMScheduler"]


BetaSchedule = Literal["scaled_linear", "linear"]
PredictionType = Literal["epsilon", "v_prediction"]
TimestepSpacing = Literal["leading", "trailing"]


def _rescale_zero_terminal_snr(betas: mx.array) -> mx.array:
    """Rescale ``betas`` so the terminal alpha cumulative product is zero.

    Implements the "zero-SNR" fix from *Common Diffusion Noise Schedules
    and Sample Steps are Flawed* (Lin et al., 2024). Ensures the
    training and inference distributions agree at the last step.
    """
    alphas = 1.0 - betas
    alphas_cumprod = mx.cumprod(alphas)
    alphas_bar_sqrt = mx.sqrt(alphas_cumprod)

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1]
    alphas_bar_sqrt = (
        (alphas_bar_sqrt - alphas_bar_sqrt_T)
        * alphas_bar_sqrt_0
        / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    )

    alphas_cumprod = alphas_bar_sqrt**2
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = mx.concatenate([alphas_cumprod[:1], alphas])
    return 1.0 - alphas


class DDIMScheduler:
    """Deterministic DDIM scheduler (``eta=0``).

    Args:
        num_train_timesteps: Number of diffusion steps used during
            training. Defaults to ``1000``.
        beta_start: First beta value. Defaults to ``0.00085`` (Stable
            Diffusion convention).
        beta_end: Last beta value. Defaults to ``0.012``.
        beta_schedule: ``"scaled_linear"`` (default) or ``"linear"``.
            ``"scaled_linear"`` matches Stable Diffusion's
            ``linspace(sqrt(b0), sqrt(bT), N)**2``.
        prediction_type: ``"epsilon"`` (vanilla) or ``"v_prediction"``
            (Salimans & Ho 2022, used by CogVideoX, SD 2.x v-pred).
        rescale_betas_zero_snr: Whether to apply zero-terminal-SNR
            rescaling to the beta schedule.
        timestep_spacing: ``"leading"`` (Stable Diffusion convention)
            or ``"trailing"`` (CogVideoX convention).
        set_alpha_to_one: Whether the final ``alpha_cumprod`` (used as
            ``prev`` when stepping past index 0) is forced to ``1.0``.
        clip_sample: Whether to clip the predicted ``x0`` to ``[-1, 1]``
            before computing the step. Set ``False`` for latent-space
            diffusion (default).
        num_inference_steps: Default schedule length. Call
            :meth:`set_timesteps` to change it later.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: BetaSchedule = "scaled_linear",
        prediction_type: PredictionType = "v_prediction",
        rescale_betas_zero_snr: bool = True,
        timestep_spacing: TimestepSpacing = "trailing",
        set_alpha_to_one: bool = True,
        clip_sample: bool = False,
        num_inference_steps: int = 50,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.clip_sample = clip_sample

        if beta_schedule == "scaled_linear":
            betas = mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "linear":
            betas = mx.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"unsupported beta_schedule: {beta_schedule!r}")

        if rescale_betas_zero_snr:
            betas = _rescale_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(alphas)
        # ``prev`` for the very first step (t = timesteps[0]) reads this.
        self.final_alpha_cumprod = mx.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.num_inference_steps = num_inference_steps
        self._timesteps: mx.array = mx.array([], dtype=mx.int32)
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Recompute the timestep schedule for ``num_inference_steps`` steps."""
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) cannot exceed "
                f"num_train_timesteps ({self.num_train_timesteps})"
            )
        self.num_inference_steps = num_inference_steps

        if self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            timesteps = (
                mx.round(mx.arange(num_inference_steps, 0, -1) * step_ratio).astype(mx.int32) - 1
            )
            timesteps = mx.clip(timesteps, 0, self.num_train_timesteps - 1)
        else:  # "leading"
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = mx.arange(0, num_inference_steps) * step_ratio
            timesteps = timesteps[::-1]

        self._timesteps = timesteps

    @property
    def timesteps(self) -> mx.array:
        """Schedule indices in iteration order (descending in t)."""
        return self._timesteps

    def _prev_timestep(self, timestep: int) -> int:
        # Look the timestep up in the actual schedule so ``prev`` is always
        # the next scheduled entry (recomputing from the step ratio drifts
        # from the rounded schedule by one for some step counts).
        schedule = [item_int(t) for t in self._timesteps]
        try:
            i = schedule.index(timestep)
        except ValueError:
            raise ValueError(
                f"timestep {timestep} is not in the current schedule; "
                f"iterate over scheduler.timesteps (or call set_timesteps())."
            ) from None
        if i + 1 < len(schedule):
            return schedule[i + 1]
        # Past the last scheduled step: extrapolate one spacing below.
        step_ratio = max(1, round(self.num_train_timesteps / self.num_inference_steps))
        return timestep - step_ratio

    def step(
        self,
        model_output: mx.array,
        timestep: int | mx.array,
        sample: mx.array,
    ) -> mx.array:
        """Run one deterministic DDIM step.

        Args:
            model_output: The model's prediction at ``timestep``
                (``epsilon`` or ``v`` depending on ``prediction_type``).
            timestep: Current step's index (int or 0-d ``mx.array``).
            sample: Current noisy sample.

        Returns:
            The denoised sample for the previous timestep.
        """
        t = item_int(timestep) if isinstance(timestep, mx.array) else int(timestep)
        prev_t = self._prev_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

        sqrt_alpha_t = mx.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_t = mx.sqrt(1.0 - alpha_prod_t)
        sqrt_alpha_t_prev = mx.sqrt(alpha_prod_t_prev)
        sqrt_one_minus_alpha_t_prev = mx.sqrt(1.0 - alpha_prod_t_prev)

        if self.prediction_type == "v_prediction":
            # v = sqrt(alpha_t) * eps - sqrt(1 - alpha_t) * x0
            pred_x0 = sqrt_alpha_t * sample - sqrt_one_minus_alpha_t * model_output
            pred_eps = sqrt_alpha_t * model_output + sqrt_one_minus_alpha_t * sample
        else:  # "epsilon"
            pred_x0 = (sample - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
            pred_eps = model_output

        if self.clip_sample:
            pred_x0 = mx.clip(pred_x0, -1, 1)

        return sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * pred_eps

    def add_noise(
        self,
        original: mx.array,
        noise: mx.array,
        timestep: int | mx.array,
    ) -> mx.array:
        """Forward-diffuse ``original`` to noise level ``timestep``."""
        t = item_int(timestep) if isinstance(timestep, mx.array) else int(timestep)
        alpha_prod_t = self.alphas_cumprod[t]
        return mx.sqrt(alpha_prod_t) * original + mx.sqrt(1.0 - alpha_prod_t) * noise
