"""Package-level API surface tests."""

import mlx_arsenal

SUBMODULES = [
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


def test_all_submodules_importable_from_root():
    # `import mlx_arsenal; mlx_arsenal.attention` must work for every
    # submodule — a partial re-export list surprises users with
    # AttributeError on half the library.
    for name in SUBMODULES:
        assert hasattr(mlx_arsenal, name), f"mlx_arsenal.{name} not re-exported"


def test_root_all_lists_submodules_and_version():
    assert sorted(mlx_arsenal.__all__) == sorted([*SUBMODULES, "__version__"])


def test_version_is_a_string():
    assert isinstance(mlx_arsenal.__version__, str) and mlx_arsenal.__version__
