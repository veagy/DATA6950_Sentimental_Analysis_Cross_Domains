#!/usr/bin/env python3
"""
Scan thesis configs, checkpoints (run_meta), training scripts, docs/logs/DUMMY indexes;
write a model documentation catalog under output/models/.

See output index.md after each run.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[3]


def parse_run_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def find_run_meta_for_file(weight_path: Path, stop_at: Path) -> Optional[Path]:
    d = weight_path.parent
    stop_at = stop_at.resolve()
    while True:
        candidate = d / "run_meta.txt"
        if candidate.is_file():
            return candidate
        if d == stop_at or d.parent == d:
            return None
        d = d.parent


def extract_docstring_summary(py_path: Path, max_lines: int = 50) -> str:
    try:
        lines = py_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    if i >= len(lines):
        return ""
    s = lines[i].strip()
    if s.startswith('"""') or s.startswith("'''"):
        quote = s[:3]
        body: list[str] = []
        first = s[3:]
        if first.endswith(quote) and len(first) > 3:
            return first[:-3].strip()
        body.append(first)
        i += 1
        while i < min(len(lines), max_lines):
            line = lines[i]
            if quote in line:
                body.append(line.split(quote, 1)[0])
                break
            body.append(line)
            i += 1
        return " ".join(x.strip() for x in body if x.strip())[:300]
    return ""


def shallow_config_extract(raw: Any) -> dict[str, Any]:
    """Pull common fields from thesis config JSON."""
    out: dict[str, Any] = {}
    if isinstance(raw, dict) and raw:
        if all(isinstance(v, dict) for v in raw.values()) and len(raw) == 1:
            class_name = next(iter(raw))
            out["model_class"] = class_name
            inner = raw[class_name]
            if isinstance(inner, dict):
                for key in (
                    "n_classes",
                    "model_name",
                    "tokenizer_name",
                    "hidden_size",
                    "num_labels",
                    "embed_dim",
                ):
                    if key in inner:
                        out[key] = inner[key]
        else:
            out["top_keys"] = list(raw.keys())[:20]
    elif isinstance(raw, list):
        out["list_length"] = len(raw)
        if raw and isinstance(raw[0], dict):
            out["list_item_keys"] = list(raw[0].keys())
    return out


def summarize_moe_manifest(raw: Any, rel_path: str) -> dict[str, Any]:
    row: dict[str, Any] = {"path": rel_path, "kind": "moe_manifest"}
    if isinstance(raw, list):
        row["n_experts"] = len(raw)
        names = []
        configs = []
        checkpoints = []
        for item in raw:
            if isinstance(item, dict):
                names.append(item.get("name", ""))
                configs.append(str(item.get("config", "")))
                checkpoints.append(str(item.get("checkpoint", "")))
        row["expert_names"] = names
        row["expert_configs"] = configs[:30]
        row["expert_checkpoints"] = checkpoints[:30]
    elif isinstance(raw, dict):
        row["keys"] = list(raw.keys())
    return row


def categorize_doc(rel: str) -> str:
    p = rel.replace("\\", "/").lower()
    if "progress" in p:
        return "progress"
    if "pretrain" in p or "hrm_encoder" in p:
        return "pretrain"
    if "train" in p or "pipeline" in p or "fine_tun" in p or "/ml/" in p:
        return "training"
    if "parameter" in p or "stack" in p or "model_param" in p:
        return "parameters"
    if "overview" in p or "inventory" in p or "config" in p:
        return "overview"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export thesis model documentation catalog.")
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: <repo>/output/models")
    ap.add_argument("--checkpoint-root", type=Path, default=None, help="Default: <repo>/checkpoints")
    ap.add_argument("--docs-root", type=Path, default=None, help="Default: <repo>/docs")
    ap.add_argument("--logs-root", type=Path, default=None, help="Default: <repo>/logs")
    ap.add_argument("--dummy-root", type=Path, default=None, help="Default: <repo>/DUMMY")
    ap.add_argument("--mirror-docs", action="store_true", help="Copy allowlisted docs into docs_mirror/")
    ap.add_argument("--max-log-files", type=int, default=500, help="Cap log file rows (newest first)")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = (args.output_dir or (repo / "output" / "models")).resolve()
    ckpt_root = (args.checkpoint_root or (repo / "checkpoints")).resolve()
    docs_root = (args.docs_root or (repo / "docs")).resolve()
    logs_root = (args.logs_root or (repo / "logs")).resolve()
    dummy_root = (args.dummy_root or (repo / "DUMMY")).resolve()
    config_root = (repo / "Code" / "thesis" / "config").resolve()
    thesis_dir = (repo / "Code" / "thesis").resolve()
    train_dir = (thesis_dir / "train").resolve()
    scripts_dir = (repo / "scripts").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # --- Config catalog (exclude moe expert JSON from main table; separate moe_manifests)
    config_rows: list[dict[str, Any]] = []
    moe_manifests: list[dict[str, Any]] = []
    for json_path in sorted(config_root.rglob("*.json")):
        rel = str(json_path.relative_to(repo))
        pnorm = rel.replace("\\", "/")
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            config_rows.append(
                {
                    "relative_path": rel,
                    "error": str(e),
                    "mtime_iso": datetime.fromtimestamp(
                        json_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
            continue
        mtime = datetime.fromtimestamp(json_path.stat().st_mtime, tz=timezone.utc).isoformat()
        if "/moe/" in pnorm and json_path.name.startswith("experts"):
            moe_manifests.append(summarize_moe_manifest(raw, rel))
            config_rows.append(
                {
                    "relative_path": rel,
                    "kind": "moe_manifest",
                    "mtime_iso": mtime,
                    **{k: v for k, v in shallow_config_extract(raw).items()},
                }
            )
            continue
        row = {
            "relative_path": rel,
            "kind": "model_config",
            "mtime_iso": mtime,
            **shallow_config_extract(raw),
        }
        row["json_top_keys"] = (
            list(raw.keys()) if isinstance(raw, dict) else f"list[{len(raw)}]"
            if isinstance(raw, list)
            else type(raw).__name__
        )
        config_rows.append(row)

    (out_dir / "configs_catalog.json").write_text(
        json.dumps(config_rows, indent=2, default=str), encoding="utf-8"
    )
    flat_keys = sorted({k for r in config_rows for k in r.keys()}) if config_rows else ["relative_path"]
    with open(out_dir / "configs_catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        for r in config_rows:
            w.writerow(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
            )

    (out_dir / "moe_manifests.json").write_text(
        json.dumps(moe_manifests, indent=2, default=str), encoding="utf-8"
    )

    # --- All run_meta.txt
    meta_paths = sorted(ckpt_root.rglob("run_meta.txt")) if ckpt_root.is_dir() else []
    jsonl_path = out_dir / "run_meta_parsed.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for mp in meta_paths:
            kv = parse_run_meta(mp)
            rec = {"run_meta_path": str(mp.relative_to(repo)), **kv}
            jf.write(json.dumps(rec, default=str) + "\n")

    # --- Checkpoint weights inventory
    weight_ext = (".safetensors", ".joblib")
    inv_rows: list[dict[str, Any]] = []
    if ckpt_root.is_dir():
        for wpath in sorted(ckpt_root.rglob("*")):
            if not wpath.is_file() or wpath.suffix not in weight_ext:
                continue
            st = wpath.stat()
            meta_path = find_run_meta_for_file(wpath, ckpt_root)
            kv: dict[str, str] = parse_run_meta(meta_path) if meta_path else {}
            inv_rows.append(
                {
                    "weight_relative": str(wpath.relative_to(repo)),
                    "size_bytes": st.st_size,
                    "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    "run_meta_relative": str(meta_path.relative_to(repo)) if meta_path else "",
                    "meta_config": kv.get("config", ""),
                    "meta_n_classes": kv.get("n_classes", ""),
                    "meta_pretrain_ckpt": kv.get("pretrain_ckpt", ""),
                    "meta_data_parquet": kv.get("data_parquet", ""),
                    "meta_layout": kv.get("layout", ""),
                }
            )
    inv_fields = [
        "weight_relative",
        "size_bytes",
        "mtime_iso",
        "run_meta_relative",
        "meta_config",
        "meta_n_classes",
        "meta_pretrain_ckpt",
        "meta_data_parquet",
        "meta_layout",
    ]
    if ckpt_root.is_dir():
        with open(out_dir / "checkpoints_inventory.csv", "w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=inv_fields)
            dw.writeheader()
            dw.writerows(inv_rows)

    # --- Training entrypoints
    train_lines = ["# Training entrypoints", "", "| Script | Summary |", "| --- | --- |"]
    if train_dir.is_dir():
        for py in sorted(train_dir.glob("train_*.py")):
            summ = extract_docstring_summary(py)
            esc = summ.replace("|", "\\|")
            train_lines.append(f"| `{py.name}` | {esc} |")
    train_lines.append("")
    (out_dir / "training_entrypoints.md").write_text("\n".join(train_lines), encoding="utf-8")

    # --- Python inventory under Code/thesis/
    code_py_rows: list[dict[str, Any]] = []
    if thesis_dir.is_dir():
        for py_path in sorted(thesis_dir.rglob("*.py")):
            try:
                text_lines = py_path.read_text(encoding="utf-8").splitlines()
                line_count = len(text_lines)
                mtime_py = datetime.fromtimestamp(
                    py_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                continue
            doc_sum = extract_docstring_summary(py_path)
            code_py_rows.append(
                {
                    "relative_path": py_path.relative_to(repo).as_posix(),
                    "line_count": line_count,
                    "mtime_iso": mtime_py,
                    "module_doc_first_line": (doc_sum[:500] if doc_sum else ""),
                }
            )
    with open(out_dir / "code_python_inventory.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.DictWriter(
            f,
            fieldnames=["relative_path", "line_count", "mtime_iso", "module_doc_first_line"],
        )
        cw.writeheader()
        cw.writerows(code_py_rows)

    # --- Validation-related entrypoints (curated globs + explicit paths)
    validation_paths: set[Path] = set()
    if thesis_dir.is_dir():
        test_dir = thesis_dir / "test"
        if test_dir.is_dir():
            for p in test_dir.rglob("*.py"):
                validation_paths.add(p.resolve())
        tools_dir = thesis_dir / "tools"
        if tools_dir.is_dir():
            for pat in ("*eval*.py", "*valid*.py", "analyze*.py"):
                for p in tools_dir.glob(pat):
                    validation_paths.add(p.resolve())
        for explicit in (
            thesis_dir / "test" / "validate_all.py",
            thesis_dir / "tools" / "eval_per_source_stem_metrics.py",
        ):
            if explicit.is_file():
                validation_paths.add(explicit.resolve())
    val_lines = ["# Validation-related entrypoints", "", "| Path | Summary |", "| --- | --- |"]
    for vp in sorted(validation_paths, key=lambda x: x.relative_to(repo).as_posix()):
        vsum = extract_docstring_summary(vp)
        esc_v = vsum.replace("|", "\\|")
        val_lines.append(f"| `{vp.relative_to(repo).as_posix()}` | {esc_v} |")
    val_lines.append("")
    (out_dir / "validation_entrypoints.md").write_text("\n".join(val_lines), encoding="utf-8")

    # --- Tools index (all tools/*.py)
    tool_lines = ["# `Code/thesis/tools/` Python index", "", "| Script | Summary |", "| --- | --- |"]
    tools_py_dir = thesis_dir / "tools"
    if tools_py_dir.is_dir():
        for tpy in sorted(tools_py_dir.glob("*.py")):
            tsum = extract_docstring_summary(tpy).replace("|", "\\|")
            tool_lines.append(f"| `{tpy.name}` | {tsum} |")
    tool_lines.append("")
    (out_dir / "code_tools_index.md").write_text("\n".join(tool_lines), encoding="utf-8")

    # --- Scripts index
    scr_lines = ["# Shell scripts under `scripts/`", ""]
    if scripts_dir.is_dir():
        for sh in sorted(scripts_dir.glob("*.sh")):
            scr_lines.append(f"## `{sh.name}`")
            scr_lines.append("")
            try:
                head = sh.read_text(encoding="utf-8").splitlines()[:20]
            except OSError:
                head = ["(read error)"]
            scr_lines.extend("    " + ln for ln in head)
            scr_lines.append("")
    (out_dir / "scripts_index.md").write_text("\n".join(scr_lines), encoding="utf-8")

    # --- Documentation sources (all .md under docs)
    doc_sections: dict[str, list[str]] = {
        "overview": [],
        "training": [],
        "pretrain": [],
        "parameters": [],
        "progress": [],
        "other": [],
    }
    if docs_root.is_dir():
        for md in sorted(docs_root.rglob("*.md")):
            rel = md.relative_to(repo).as_posix()
            cat = categorize_doc(rel)
            if cat not in doc_sections:
                cat = "other"
            doc_sections[cat].append(f"- `{rel}`")
    doc_buf = ["# Documentation sources (thesis models and training)", "", "Paths relative to repository root.", ""]
    for cat, items in doc_sections.items():
        if not items:
            continue
        doc_buf.append(f"## {cat}")
        doc_buf.extend(sorted(items))
        doc_buf.append("")
    (out_dir / "documentation_sources.md").write_text("\n".join(doc_buf), encoding="utf-8")

    # --- Logs index
    log_rows: list[dict[str, Any]] = []
    if logs_root.is_dir():
        for lp in logs_root.rglob("*"):
            if not lp.is_file():
                continue
            st = lp.stat()
            log_rows.append(
                {
                    "path": str(lp.relative_to(repo)),
                    "size_bytes": st.st_size,
                    "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
    log_rows.sort(key=lambda r: r["mtime_iso"], reverse=True)
    if args.max_log_files > 0:
        log_rows = log_rows[: args.max_log_files]
    if logs_root.is_dir():
        with open(out_dir / "logs_index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "mtime_iso"])
            w.writeheader()
            w.writerows(log_rows)

    # --- DUMMY smoke index
    dummy_lines = ["# DUMMY smoke / mock artifacts", "", f"Root: `{dummy_root.relative_to(repo) if dummy_root.is_dir() else 'missing'}`", ""]
    if dummy_root.is_dir():
        seen: set[Path] = set()
        allow_ext = frozenset({".sh", ".py", ".log", ".md", ".txt", ".bat", ".ps1", ".csv", ".json"})
        for p in sorted(dummy_root.rglob("*")):
            if not p.is_file() or p.resolve() in seen:
                continue
            if p.stat().st_size > 10_000_000:
                continue
            suff = p.suffix.lower()
            if suff not in allow_ext:
                continue
            rel_d = p.relative_to(dummy_root).as_posix()
            if suff == ".csv" and "eval" not in rel_d.lower() and "outputs" not in rel_d.lower():
                continue
            seen.add(p.resolve())
            dummy_lines.append(f"- `{p.relative_to(repo).as_posix()}` ({p.stat().st_size} bytes)")
        dummy_lines.append("")
    (out_dir / "dummy_smoke_index.md").write_text("\n".join(dummy_lines), encoding="utf-8")

    # --- Optional docs mirror
    mirror_note = ""
    if args.mirror_docs:
        mirror_dir = out_dir / "docs_mirror"
        if mirror_dir.exists():
            shutil.rmtree(mirror_dir)
        mirror_dir.mkdir(parents=True)
        allow = [
            "docs/thesis_config_inventory.md",
            "docs/models_overview.md",
            "docs/Model_Parameters_and_Stacking.md",
            "docs/thesis_parameter_counts.md",
            "docs/ml/TRAINING_PIPELINES.md",
            "docs/Fine_tuning_and_data_pipeline_implementation_summary.md",
            "docs/hrm_encoder_pretrain_runbook.md",
            "docs/ml/README.md",
        ]
        for rel in allow:
            src = repo / rel
            if not src.is_file():
                continue
            sub = Path(rel).relative_to("docs")
            dest = mirror_dir / sub
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        mirror_note = f"- **Docs mirror:** `{mirror_dir.relative_to(repo)}` (allowlisted copies)\n"

    # --- index.md
    index = [
        "# Model documentation catalog",
        "",
        f"- **Generated (UTC):** {now}",
        f"- **Repository root:** `{repo}`",
        "",
        "## Outputs in this folder",
        "",
        "- [`configs_catalog.csv`](configs_catalog.csv) / [`configs_catalog.json`](configs_catalog.json) — one row per `Code/thesis/config/**/*.json`",
        "- [`moe_manifests.json`](moe_manifests.json) — MoE expert manifest summaries",
        "- [`checkpoints_inventory.csv`](checkpoints_inventory.csv) — weight files + joined `run_meta.txt` fields",
        "- [`run_meta_parsed.jsonl`](run_meta_parsed.jsonl) — every `run_meta.txt` under checkpoints",
        "- [`training_entrypoints.md`](training_entrypoints.md) — `train_*.py` scripts",
        "- [`code_python_inventory.csv`](code_python_inventory.csv) — every `*.py` under `Code/thesis/`",
        "- [`validation_entrypoints.md`](validation_entrypoints.md) — test + eval/validation tooling scripts",
        "- [`code_tools_index.md`](code_tools_index.md) — `Code/thesis/tools/*.py` discovery table",
        "- [`scripts_index.md`](scripts_index.md) — `scripts/*.sh` headers",
        "- [`documentation_sources.md`](documentation_sources.md) — index of `docs/**/*.md` by category",
        "- [`logs_index.csv`](logs_index.csv) — log files (newest first, capped)",
        "- [`dummy_smoke_index.md`](dummy_smoke_index.md) — DUMMY smoke paths",
        mirror_note,
        "## Counts",
        "",
        f"- Config JSON files: **{len(config_rows)}**",
        f"- MoE manifest summaries: **{len(moe_manifests)}**",
        f"- `run_meta.txt` files: **{len(meta_paths)}**",
        f"- Checkpoint weights (`*.safetensors`, `*.joblib`): **{len(inv_rows)}**",
        f"- Log files indexed: **{len(log_rows)}**",
        f"- Python modules under `Code/thesis/`: **{len(code_py_rows)}**",
        f"- Validation entrypoint scripts listed: **{len(validation_paths)}**",
        "",
    ]
    index_clean = [x for x in index if x is not None]
    (out_dir / "index.md").write_text("\n".join(index_clean), encoding="utf-8")

    print(f"[export_model_documentation] wrote catalog under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
