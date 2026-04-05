# Transformer MLP finetune smoke test — implementation summary

**Last updated:** 2026-04-02

**Related:** Full transformer MLP feature summary: [`transformer_mlp_head_finetune_implementation.md`](transformer_mlp_head_finetune_implementation.md)  
**Parity reference:** HRM smoke: [`hrm_mlp_finetune_smoke_implementation.md`](hrm_mlp_finetune_smoke_implementation.md)

---

## Purpose

Provide a **sandbox** run before full-dataset finetune: same launcher and training code as production, but **~1000 rows** (`THESIS_MAX_SAMPLES`), **isolated I/O** under `DUMMY/transformer-mlp-finetune/`, aggressive **periodic resume** saves, and a repeatable path to test **interrupt + restore**. Does **not** write under production `checkpoints/` (unless you override `THESIS_CHECKPOINT_ROOT`).

---

## Mechanisms (shared with HRM smoke)

| Mechanism | Location / behavior |
|-----------|---------------------|
| Row cap | `ParquetTextDataset` + `--max_samples` / `THESIS_MAX_SAMPLES` |
| Resume root | `--resume_temp_root` / `THESIS_RESUME_TEMP`; `LiveResumeDir.try_restore_training` |
| Periodic saves | `THESIS_SAVE_EVERY_STEPS`, `THESIS_SAVE_EVERY_MINUTES` in `train_single.py` |
| Production launcher | `scripts/run_transformer_mlp_finetune.sh` — smoke **exec**s this script |

---

## Delivered artifacts

| Artifact | Path | Role |
|----------|------|------|
| Main smoke runner | `DUMMY/transformer-mlp-finetune/run_transformer_mlp_finetune_smoke.sh` | Sandbox env, preflight parquet, calls `scripts/run_transformer_mlp_finetune.sh` |
| Interrupt demo | `DUMMY/transformer-mlp-finetune/demo_resume_interrupt.sh` | `timeout -s INT` (default 120s) then second smoke run; defaults **single GPU** |
| Logs | `DUMMY/transformer-mlp-finetune/logs/` | `transformer_finetune_smoke_{2,3}label.log` |
| Live resume | `DUMMY/transformer-mlp-finetune/resume/` | Slugged subdirs from training |
| Checkpoints | `DUMMY/transformer-mlp-finetune/checkpoints/` | `{K}-labels/all-data/*.safetensors` |

---

## Prerequisites

| Requirement | Note |
|-------------|------|
| Parquet | `${THESIS_DATA_ROOT:-data}/processed/${THESIS_TRANSFORMER_DATASET_STEM:-all-data}.parquet` |
| HF weights | Downloaded/cached under `LLMModule` `checkpoint_dir` (e.g. `checkpoints/deep_learning/llm` relative to cwd) on first run |
| GPU | Same as production; override batch with `THESIS_TRANSFORMER_FINETUNE_BATCH` |

---

## Default environment (smoke script)

| Variable | Default | Purpose |
|----------|---------|---------|
| `THESIS_MAX_SAMPLES` | `1000` | Cap training rows |
| `THESIS_CHECKPOINT_ROOT` | `$ROOT/DUMMY/transformer-mlp-finetune/checkpoints` | Sandbox checkpoints |
| `THESIS_RESUME_TEMP` | `$ROOT/DUMMY/transformer-mlp-finetune/resume` | Sandbox resume |
| `THESIS_TRANSFORMER_FINETUNE_LOG` | `DUMMY/transformer-mlp-finetune/logs/transformer_finetune_smoke_${LABELS}label.log` | Tee log |
| `THESIS_EPOCHS_FINETUNE` | `3` | Enough epochs to see periodic saves |
| `THESIS_SAVE_EVERY_STEPS` | `15` | Frequent step-based snapshots |
| `THESIS_SAVE_EVERY_MINUTES` | `1` | Frequent time-based snapshots |
| `THESIS_DETACH` | `none` | Foreground smoke |
| `THESIS_NUM_WORKERS` | `4` | DataLoader workers |
| `THESIS_TRANSFORMER_CFG_STEM` | `B3_E_DL1_DistilBERT_mlp768_1024` | Which MLP config to smoke |

---

## How to run

From **repo root** (directory containing `Code/` and `DUMMY/`):

```bash
# 3-class smoke (default 1000 rows)
bash DUMMY/transformer-mlp-finetune/run_transformer_mlp_finetune_smoke.sh

# 2-class smoke
bash DUMMY/transformer-mlp-finetune/run_transformer_mlp_finetune_smoke.sh 2

# Another MLP config
THESIS_TRANSFORMER_CFG_STEM=B4_E_DL3_BERT_mlp768_1024 bash DUMMY/transformer-mlp-finetune/run_transformer_mlp_finetune_smoke.sh 3

# Alternate processed tree
THESIS_DATA_ROOT="$ROOT/BACKUP/data" bash DUMMY/transformer-mlp-finetune/run_transformer_mlp_finetune_smoke.sh
```

**Interrupt + resume demo** (recommended single-GPU):

```bash
bash DUMMY/transformer-mlp-finetune/demo_resume_interrupt.sh       # 3-class
bash DUMMY/transformer-mlp-finetune/demo_resume_interrupt.sh 2       # 2-class
# FIRST_PHASE_SEC=90 bash DUMMY/transformer-mlp-finetune/demo_resume_interrupt.sh
```

**Multi-GPU:** run smoke normally; for interrupt testing, prefer **manual Ctrl+C** then re-run (avoid `timeout` across ranks).

---

## One-step forward smoke (optional)

```bash
cd /path/to/repo
python3 Code/thesis/tools/transformer_mlp_head_smoke.py \
  --config Code/thesis/config/transformers/2_labels/B3_E_DL1_DistilBERT_mlp768_1024.json \
  --n-classes 2
```

Requires PyTorch + transformers installed in that interpreter.

---

## Verification checklist

1. Log file under `DUMMY/transformer-mlp-finetune/logs/`.
2. Periodic resume artifacts under `DUMMY/transformer-mlp-finetune/resume/` during the run.
3. Final `*.safetensors` under `DUMMY/transformer-mlp-finetune/checkpoints/{2,3}-labels/all-data/`.
4. After interrupt demo: second run **restores** when metadata matches.

---

## Cleanup

```bash
rm -rf DUMMY/transformer-mlp-finetune/logs DUMMY/transformer-mlp-finetune/resume DUMMY/transformer-mlp-finetune/checkpoints
```

---

## Out of scope

- Smoke does not replace full `all-data` training; omit `THESIS_MAX_SAMPLES` for production-length runs.
- No dedicated pytest; validation is operational (bash + GPU).
