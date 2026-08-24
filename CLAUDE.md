# CLAUDE.md

Project guidance for Claude Code working in this repo.

## What this is

`mlx-arsenal` is a library of mid-level building blocks for [Apple MLX](https://github.com/ml-explore/mlx) — the missing layer between `mlx.nn` and full model implementations. Modules cover diffusion primitives, spatial ops, attention masks, MoE routing, channel-layout conversion, weight loading, rasterization, and tiling.

When porting PyTorch models to MLX, prefer functions from `mlx_arsenal.*` over hand-rolling.

## Layout

- `src/mlx_arsenal/` — `src/` package layout (migrated from flat layout in a106fa0). Submodules: `attention`, `conv`, `diffusion`, `encoding`, `ffn`, `layout`, `loader`, `modulation`, `moe`, `norm`, `rasterize`, `rope`, `spatial`, `streaming`, `tiling`.
- `tests/` — pytest tests, one file per submodule (`test_<module>.py`).
- `docs/` — design notes and ADRs.

## Conventions

- **Tensor layout:** channels-last (NHWC / NDHWC) by default — MLX-native. Conversion helpers live in `mlx_arsenal.layout`.
- **Dim names in code:** uppercase single-letters (`B`, `C`, `H`, `W`, `T`, `D`, `L`) are idiomatic — `ruff` rule `N806` is disabled for this reason.
- **Type hints:** required (`PYTHON_QU003` strict mode via `ty`). Use `T | None` not `Optional[T]`. Prefer `from collections.abc` for `Callable`, `Sequence`, etc.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) — enforced by `commitlint` (config-conventional / Angular) in CI. Allowed types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`. Release commits are emitted by release-please as `chore(main): release X.Y.Z` (special-cased in `commitlint.config.cjs`) — never write them by hand.
- **Versioning:** SemVer 2.0.0. Releases are cut via `release-please`.

## Workflow

- Install dev: `pip install -e ".[dev]"` (or `uv sync --extra dev`).
- Lint: `ruff check src/mlx_arsenal tests` + `ruff format --check ...`
- Type-check: `ty check`
- Tests: `pytest tests/ -v`
- Pre-commit hooks are wired up — install with `pre-commit install`.

## Governance

Repo follows [Intendant](https://github.com/dgrauet/intendant) governance (`.intendant.toml`, advisory mode). Run `intendant audit` (or via the MCP server) before merging structural changes.

## Related skills

- `mlx-porting` — for translating PyTorch / CUDA diffusion / transformer code to MLX. Always invoke when working on a port.
