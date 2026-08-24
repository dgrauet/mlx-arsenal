"""mlx-arsenal: Reusable mid-level building blocks for MLX."""

from importlib.metadata import PackageNotFoundError, version

from . import attention as attention
from . import conv as conv
from . import diffusion as diffusion
from . import encoding as encoding
from . import ffn as ffn
from . import layout as layout
from . import loader as loader
from . import modulation as modulation
from . import moe as moe
from . import norm as norm
from . import rasterize as rasterize
from . import rope as rope
from . import spatial as spatial
from . import streaming as streaming
from . import tiling as tiling

try:
    __version__ = version("mlx-arsenal")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "attention",
    "conv",
    "diffusion",
    "encoding",
    "ffn",
    "layout",
    "loader",
    "modulation",
    "moe",
    "norm",
    "rasterize",
    "rope",
    "spatial",
    "streaming",
    "tiling",
]
