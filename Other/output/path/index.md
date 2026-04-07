# TEMP path inventory

- **Generated (UTC):** 2026-04-05T04:43:20.097037+00:00
- **Repository root:** `D:\CAPSTONE\new\TEMP`

## Run

- **Default excludes disabled:** False
- **Exclude set (directory name matches any path component):** '.git', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.venv', '__pycache__', 'checkpoints', 'data', 'logs', 'node_modules', 'output', 'venv'

## Outputs

- [`inventory_all.csv`](inventory_all.csv) — **2117** rows (**1473** files, **644** directories)
- [`summary_by_topdir.md`](summary_by_topdir.md) — counts and file-byte totals by first path segment

Full-tree runs without default excludes can be very large and slow if `data/`, `checkpoints/`, or `output/` are present.
