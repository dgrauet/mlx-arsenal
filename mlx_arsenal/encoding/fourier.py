import mlx.core as mx
import mlx.nn as nn


class FourierEmbedder(nn.Module):
    """Fourier positional encoding for continuous coordinates.

    Maps input coordinates to a higher-dimensional space using sinusoidal
    functions at multiple frequencies. Commonly used for encoding 3D positions
    in neural fields (NeRF, SDF networks).

    Reference: Hunyuan3D-2.1 autoencoders/model.py FourierEmbedder

    Args:
        num_freqs: Number of frequency bands.
        input_dim: Dimensionality of input coordinates.
        include_pi: Whether to multiply frequencies by pi.
        include_input: Whether to concatenate raw input to output.
    """

    def __init__(
        self,
        num_freqs: int = 6,
        input_dim: int = 3,
        include_pi: bool = True,
        include_input: bool = True,
    ):
        super().__init__()
        self.include_input = include_input
        frequencies = 2.0 ** mx.arange(num_freqs)
        if include_pi:
            frequencies = frequencies * mx.array(3.141592653589793)
        self.frequencies = frequencies
        self.out_dim = input_dim * (2 * num_freqs + (1 if include_input else 0))

    def __call__(self, x: mx.array) -> mx.array:
        """Encode coordinates with Fourier features.

        Args:
            x: Input coordinates of shape (..., input_dim).

        Returns:
            Fourier features of shape (..., out_dim).
        """
        # x: (..., D), frequencies: (F,)
        # embed: (..., D, F)
        embed = x[..., None] * self.frequencies
        # Flatten last two dims: (..., D*F)
        embed = embed.reshape(*x.shape[:-1], -1)
        parts = []
        if self.include_input:
            parts.append(x)
        parts.append(mx.sin(embed))
        parts.append(mx.cos(embed))
        return mx.concatenate(parts, axis=-1)
