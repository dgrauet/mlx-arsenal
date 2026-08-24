# Install

## Requirements

- Python ≥ 3.10
- [MLX](https://github.com/ml-explore/mlx) ≥ 0.32.1
- Apple Silicon Mac (M1 / M2 / M3 / M4)

## From PyPI

```bash
pip install mlx-arsenal
```

Or with [`uv`](https://github.com/astral-sh/uv):

```bash
uv add mlx-arsenal
```

## From source

```bash
git clone https://github.com/dgrauet/mlx-arsenal
cd mlx-arsenal
pip install -e ".[dev]"
```

Run the test suite to verify the install:

```bash
pytest tests/ -v
```

## Building these docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
