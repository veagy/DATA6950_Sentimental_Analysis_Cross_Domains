#!/usr/bin/env python3
"""
Walk the repository tree and write a CSV + index under output/path/.

Default excludes skip large or regenerated subtrees; use --no-default-excludes for a full tree.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

DEFAULT_EXCLUDE_DIR_NAMES = frozenset(
    {
        "output",
        "checkpoints",
        "data",
        "logs",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def _rel_excluded(rel: Path, exclude: frozenset[str]) -> bool:
    for part in rel.parts:
        if part in (".", ""):
            continue
        if part in exclude:
            return True
    return False


def walk_tree(
    root: Path,
    repo: Path,
    exclude: frozenset[str],
    rows: list[dict[str, str | int]],
) -> None:
    rel = root.relative_to(repo)
    if _rel_excluded(rel, exclude):
        return
    try:
        st = root.stat()
    except OSError:
        return
    mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    if root.is_dir():
        rows.append(
            {
                "relative_path": "." if rel == Path(".") else rel.as_posix(),
                "kind": "dir",
                "size_bytes": 0,
                "mtime_iso": mtime_iso,
                "suffix": "",
            }
        )
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return
        for child in children:
            walk_tree(child, repo, exclude, rows)
    elif root.is_file():
        rows.append(
            {
                "relative_path": rel.as_posix(),
                "kind": "file",
                "size_bytes": st.st_size,
                "mtime_iso": mtime_iso,
                "suffix": root.suffix.lower() or "",
            }
        )


def summary_by_topdir(rows: list[dict[str, str | int]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    bytes_by: dict[str, int] = defaultdict(int)
    for r in rows:
        rp = str(r["relative_path"])
        if rp == ".":
            top = "."
        else:
            top = rp.split("/", 1)[0]
        counts[top] += 1
        if r["kind"] == "file":
            bytes_by[top] += int(r["size_bytes"])
    lines = [
        "# Summary by top-level segment",
        "",
        "| Segment | Rows (files+dirs) | File bytes (sum) |",
        "| --- | ---: | ---: |",
    ]
    for seg in sorted(counts.keys(), key=lambda s: (s != ".", s.lower())):
        lines.append(f"| `{seg}` | {counts[seg]} | {bytes_by[seg]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export recursive filesystem inventory of the repo (default excludes for speed)."
    )
    ap.add_argument("--repo-root", type=Path, default=_REPO, help="Repository root (default: TEMP)")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/output/path)",
    )
    ap.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="NAME",
        help="Directory name to exclude anywhere in the relative path (repeatable)",
    )
    ap.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not apply built-in excludes (output, checkpoints, data, logs, .git, etc.)",
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = (args.output_dir or (repo / "output" / "path")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_set = set(args.exclude_dir)
    if not args.no_default_excludes:
        exclude_set |= set(DEFAULT_EXCLUDE_DIR_NAMES)
    exclude = frozenset(exclude_set)

    rows: list[dict[str, str | int]] = []
    walk_tree(repo, repo, exclude, rows)

    now = datetime.now(timezone.utc).isoformat()
    fieldnames = ["relative_path", "kind", "size_bytes", "mtime_iso", "suffix"]
    inv_path = out_dir / "inventory_all.csv"
    with open(inv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary_path = out_dir / "summary_by_topdir.md"
    summary_path.write_text(summary_by_topdir(rows), encoding="utf-8")

    n_files = sum(1 for r in rows if r["kind"] == "file")
    n_dirs = sum(1 for r in rows if r["kind"] == "dir")
    exclude_display = sorted(exclude) if exclude else ["(none)"]
    idx_lines = [
        "# TEMP path inventory",
        "",
        f"- **Generated (UTC):** {now}",
        f"- **Repository root:** `{repo}`",
        "",
        "## Run",
        "",
        f"- **Default excludes disabled:** {bool(args.no_default_excludes)}",
        f"- **Exclude set (directory name matches any path component):** {', '.join(repr(x) for x in exclude_display)}",
        "",
        "## Outputs",
        "",
        f"- [`inventory_all.csv`](inventory_all.csv) — **{len(rows)}** rows (**{n_files}** files, **{n_dirs}** directories)",
        f"- [`summary_by_topdir.md`](summary_by_topdir.md) — counts and file-byte totals by first path segment",
        "",
        "Full-tree runs without default excludes can be very large and slow if `data/`, `checkpoints/`, or `output/` are present.",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(idx_lines), encoding="utf-8")

    print(f"[export_temp_path_inventory] wrote {len(rows)} rows under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
