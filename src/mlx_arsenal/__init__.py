"""mlx-arsenal: Reusable mid-level building blocks for MLX."""

from importlib.metadata import PackageNotFoundError, version

from . import diffusion as diffusion
from . import encoding as encoding
from . import moe as moe
from . import rasterize as rasterize

try:
    __version__ = version("mlx-arsenal")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
