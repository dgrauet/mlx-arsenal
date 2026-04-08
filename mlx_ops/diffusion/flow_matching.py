"""Flow matching scheduler for diffusion models.

Implements the rectified flow / flow matching framework used in
Stable Diffusion 3, FLUX, and modern video generation models.
"""

from typing import Optional

import mlx.core as mx


class FlowMatchEulerScheduler:
    """Euler scheduler for flow matching (rectified flow).

    In flow matching, the forward process is a linear interpolation:
        x_t = (1 - t) * x_0 + t * noise

    The model predicts the velocity v = noise - x_0, and we solve
    the ODE with Euler steps.

    Args:
        num_inference_steps: Number of denoising steps.
        shift: Timestep shift for non-uniform spacing. 1.0 = uniform.
    """

    def __init__(self, num_inference_steps: int = 50, shift: float = 1.0):
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        self._timesteps = None
        self._sigmas = None
        self._setup()

    def _setup(self):
        # Linear spacing from 1.0 to 0.0
        sigmas = mx.linspace(1.0, 0.0, self.num_inference_steps + 1)

        # Apply shift: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

        self._sigmas = sigmas
        # Timesteps are just sigma * 1000 for model input convention
        self._timesteps = sigmas[:-1] * 1000.0

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    def scale_noise(
        self,
        sample: mx.array,
        noise: mx.array,
        sigma: float,
    ) -> mx.array:
        """Create noisy sample via linear interpolation.

        Args:
            sample: Clean sample x_0.
            noise: Random noise.
            sigma: Noise level in [0, 1].

        Returns:
            Noisy sample x_t.
        """
        return (1.0 - sigma) * sample + sigma * noise

    def step(
        self,
        model_output: mx.array,
        step_index: int,
        sample: mx.array,
    ) -> mx.array:
        """Perform one Euler step of the reverse ODE.

        Args:
            model_output: Model prediction (velocity).
            step_index: Current step index.
            sample: Current noisy sample x_t.

        Returns:
            Denoised sample x_{t-1}.
        """
        sigma = self._sigmas[step_index]
        sigma_next = self._sigmas[step_index + 1]
        dt = sigma_next - sigma
        return sample + dt * model_output

    def add_noise(
        self,
        original: mx.array,
        noise: mx.array,
        step_index: int,
    ) -> mx.array:
        """Add noise at a specific step level.

        Args:
            original: Clean sample.
            noise: Random noise.
            step_index: Step index determining noise level.

        Returns:
            Noisy sample.
        """
        sigma = float(self._sigmas[step_index])
        return self.scale_noise(original, noise, sigma)
