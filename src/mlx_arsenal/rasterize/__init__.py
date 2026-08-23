"""Tile-binned triangle rasterizer with exact fixed-point coverage, plus interpolation."""

from .interpolate import interpolate
from .rasterize import rasterize_triangles

__all__ = ["rasterize_triangles", "interpolate"]
