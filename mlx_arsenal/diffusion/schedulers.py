"""Sigma schedules and noise schedulers for flow-matching diffusion."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def get_sampling_sigmas(num_steps: int, shift: float = 1.0) -> list[float]:
    """Flow-matching sigma schedule: linspace(1, 0, steps+1) with optional shift.

    The schedule includes the terminal ``0.0`` so pairs are formed via
    ``zip(sigmas[:-1], sigmas[1:])``.

    Args:
        num_steps: Number of denoising steps.
        shift: Shift factor applied as ``shift * s / (1 + (shift - 1) * s)``.

    Returns:
        List of ``num_steps + 1`` sigma values descending from 1.0 to 0.0.
    """
    sigmas = np.linspace(1.0, 0.0, num_steps + 1)
    if shift != 1.0:
        nonzero = sigmas != 0
        shifted = sigmas.copy()
        shifted[nonzero] = shift * sigmas[nonzero] / (1.0 + (shift - 1.0) * sigmas[nonzero])
        sigmas = shifted
    return sigmas.tolist()


def dynamic_shift_schedule(
    num_steps: int,
    num_tokens: int,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
    base_tokens: int = 1024,
    max_tokens: int = 4096,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """Sigma schedule with token-count-dependent shift (LTX-style).

    Interpolates a shift factor linearly between ``base_shift`` at
    ``base_tokens`` and ``max_shift`` at ``max_tokens``, then applies it to
    a descending linspace. Optional terminal stretching matches the last
    non-zero sigma to ``1 - terminal``.

    Args:
        num_steps: Number of denoising steps.
        num_tokens: Number of latent tokens (drives the shift).
        base_shift: Shift at ``base_tokens``.
        max_shift: Shift at ``max_tokens``.
        base_tokens: Anchor for ``base_shift``.
        max_tokens: Anchor for ``max_shift``.
        stretch: Rescale so the last non-zero sigma equals ``1 - terminal``.
        terminal: Target terminal for stretching.

    Returns:
        List of ``num_steps + 1`` sigma values ending at ``0.0``.
    """
    sigmas = np.linspace(1.0, 0.0, num_steps + 1)

    slope = (max_shift - base_shift) / (max_tokens - base_tokens)
    intercept = base_shift - slope * base_tokens
    sigma_shift = num_tokens * slope + intercept

    nonzero = sigmas != 0
    shifted = np.empty_like(sigmas)
    shifted[~nonzero] = 0.0
    shifted[nonzero] = math.exp(sigma_shift) / (
        math.exp(sigma_shift) + (1.0 / sigmas[nonzero] - 1.0)
    )
    sigmas = shifted

    if stretch:
        nz = sigmas != 0
        nz_sigmas = sigmas[nz]
        if len(nz_sigmas) > 0:
            one_minus_z = 1.0 - nz_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            if scale_factor != 0:
                sigmas[nz] = 1.0 - (one_minus_z / scale_factor)

    return sigmas.tolist()


class FlowMatchEulerScheduler:
    """Stateful flow-matching Euler scheduler.

    Mirrors the diffusers ``FlowMatchEulerDiscreteScheduler`` API: ``set_timesteps``
    then step through timesteps calling ``step`` with the model's velocity output.
    ``add_noise`` forward-interpolates a clean sample toward pure noise.

    Convention: sigmas ascend from ~0 (clean) to 1 (noise), with a terminal
    ``1.0`` appended so the final step has a valid ``sigma_next``.

    Args:
        num_train_timesteps: Number of training timesteps.
        shift: Shift factor applied to the sigma schedule.
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigmas: mx.array | None = None
        self.timesteps: mx.array | None = None
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    def set_timesteps(
        self, num_inference_steps: int, sigmas: np.ndarray | list[float] | None = None
    ) -> None:
        """Configure the scheduler for ``num_inference_steps`` denoising steps.

        Args:
            num_inference_steps: Number of denoising steps.
            sigmas: Optional custom sigmas (ascending, ``[0,1]``). If omitted,
                a linspace schedule is generated.
        """
        if sigmas is None:
            timesteps = np.linspace(
                1, self.num_train_timesteps, num_inference_steps, dtype=np.float32
            )
            sigmas = timesteps / self.num_train_timesteps
        else:
            sigmas = np.asarray(sigmas, dtype=np.float32)

        sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        timesteps = sigmas * self.num_train_timesteps

        self.timesteps = mx.array(timesteps, dtype=mx.float32)
        self.sigmas = mx.array(
            np.concatenate([sigmas, np.ones(1, dtype=np.float32)]), dtype=mx.float32
        )
        self.num_inference_steps = num_inference_steps
        self._step_index = None

    def step(self, model_output: mx.array, timestep: mx.array, sample: mx.array) -> mx.array:
        """Advance the sample by one Euler step using a velocity prediction.

        Args:
            model_output: Predicted velocity (same shape as ``sample``).
            timestep: Current timestep (only used to lazy-init the step index).
            sample: Current noisy sample.

        Returns:
            Updated sample after one Euler step.
        """
        del timestep
        if self.sigmas is None:
            raise RuntimeError("set_timesteps() must be called before step()")
        if self._step_index is None:
            self._step_index = 0

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return prev_sample

    def add_noise(self, original: mx.array, noise: mx.array, sigma: mx.array) -> mx.array:
        """Flow-matching interpolation: ``sigma * noise + (1 - sigma) * original``."""
        return sigma * noise + (1.0 - sigma) * original
