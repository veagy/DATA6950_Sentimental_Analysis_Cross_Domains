# Model documentation catalog

- **Generated (UTC):** 2026-04-05T04:43:19.832851+00:00
- **Repository root:** `D:\CAPSTONE\new\TEMP`

## Outputs in this folder

- [`configs_catalog.csv`](configs_catalog.csv) / [`configs_catalog.json`](configs_catalog.json) — one row per `Code/thesis/config/**/*.json`
- [`moe_manifests.json`](moe_manifests.json) — MoE expert manifest summaries
- [`checkpoints_inventory.csv`](checkpoints_inventory.csv) — weight files + joined `run_meta.txt` fields
- [`run_meta_parsed.jsonl`](run_meta_parsed.jsonl) — every `run_meta.txt` under checkpoints
- [`training_entrypoints.md`](training_entrypoints.md) — `train_*.py` scripts
- [`code_python_inventory.csv`](code_python_inventory.csv) — every `*.py` under `Code/thesis/`
- [`validation_entrypoints.md`](validation_entrypoints.md) — test + eval/validation tooling scripts
- [`code_tools_index.md`](code_tools_index.md) — `Code/thesis/tools/*.py` discovery table
- [`scripts_index.md`](scripts_index.md) — `scripts/*.sh` headers
- [`documentation_sources.md`](documentation_sources.md) — index of `docs/**/*.md` by category
- [`logs_index.csv`](logs_index.csv) — log files (newest first, capped)
- [`dummy_smoke_index.md`](dummy_smoke_index.md) — DUMMY smoke paths

## Counts

- Config JSON files: **75**
- MoE manifest summaries: **6**
- `run_meta.txt` files: **25**
- Checkpoint weights (`*.safetensors`, `*.joblib`): **87**
- Log files indexed: **28**
- Python modules under `Code/thesis/`: **52**
- Validation entrypoint scripts listed: **3**
