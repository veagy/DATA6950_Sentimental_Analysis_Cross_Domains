# HRM MLP finetune smoke test — implementation summary

**Last updated:** 2026-03-31  

**Plan reference (read-only, local Cursor path):** `c:\Users\ravul\.cursor\plans\hrm_mlp_finetune_smoke_36c91b81.plan.md`  

**Related:** Full MLP-head finetune feature summary: [`docs/summary/hrm_mlp_head_finetune_implementation.md`](hrm_mlp_head_finetune_implementation.md)

---

## Purpose

Provide a **sandbox** run on Ubuntu + NVIDIA **before** full-dataset finetune: same launcher and training code as production, but **~1000 rows** (`THESIS_MAX_SAMPLES`), **isolated I/O** under `DUMMY/fine-tune/`, aggressive **periodic resume** saves, and a repeatable path to test **interrupt + restore** (live resume bundles). Validates checkpoint writing, resume metadata, and second-run continuation without touching `checkpoints/hrm/` or production `logs/resume/`.

---

## Codebase dependencies (no duplicate `torchrun` line)

| Mechanism | Location / behavior |
|-----------|---------------------|
| Row cap | `ParquetTextDataset` uses `df.head(max_samples)` when `--max_samples` / `THESIS_MAX_SAMPLES` is set (`Code/thesis/common/datasets.py`). |
| Resume root | `--resume_temp_root` via env `THESIS_RESUME_TEMP`; `LiveResumeDir.try_restore_training` when metadata matches (`Code/thesis/common/resume_checkpoint.py`). |
| Periodic saves | `THESIS_SAVE_EVERY_STEPS`, `THESIS_SAVE_EVERY_MINUTES` (and related knobs) in `Code/thesis/train/train_single.py` (`_maybe_periodic_resume_save`). |
| Interrupt save | `InterruptSave` path in `train_single` on SIGINT between steps (same resume module family). |
| Production launcher | `scripts/run_hrm_finetune_mlp.sh` — smoke script sets env and **exec**s this script so CUDA, batch tiers, and `torch.distributed.run` stay consistent. |

---

## Delivered artifacts

| Artifact | Path | Role |
|----------|------|------|
| Main smoke runner | `DUMMY/fine-tune/run_hrm_mlp_finetune_smoke.sh` | Sets sandbox env, preflight parquet + encoder, calls `scripts/run_hrm_finetune_mlp.sh` with label `2` or `3`. |
| Interrupt demo | `DUMMY/fine-tune/demo_resume_interrupt.sh` | `timeout -s INT` (default 120s) then second smoke run; defaults **single GPU** (`CUDA_VISIBLE_DEVICES=0`, `THESIS_NPROC_PER_NODE=1`). |
| Logs (created on run) | `DUMMY/fine-tune/logs/` | Tee log files `hrm_finetune_smoke_2label.log` / `hrm_finetune_smoke_3label.log`. |
| Live resume (created on run) | `DUMMY/fine-tune/resume/` | Slugged subdirs from training; periodic + interrupt bundles. |
| Fine-tune checkpoints (created on run) | `DUMMY/fine-tune/checkpoints/hrm/fine-tune/all-data/{K}-labels/*.safetensors` | Same layout flag as production launcher (`--hrm_finetune_checkpoint_layout` / `THESIS_HRM_FINETUNE_CKPT_LAYOUT`). |

Directories under `DUMMY/fine-tune/` are **created by the smoke script** on first run (`mkdir -p`).

---

## Encoder and data prerequisites

| Requirement | Default / note |
|-------------|----------------|
| Parquet | `${THESIS_DATA_ROOT:-data}/processed/${THESIS_HRM_PRETRAIN_STEM:-all-data}.parquet` must exist relative to repo root (`THESIS_DATA_ROOT` defaults to `$ROOT/data`). |
| Encoder | If `THESIS_HRM_ENCODER_CKPT` is **unset**, smoke sets it to **`checkpoints/hrm/pretrain/{stem}/E_HRM1_4Level.safetensors`** (production pretrain path). The sandbox **`THESIS_CHECKPOINT_ROOT`** does **not** contain the encoder; only finetune outputs go there. |
| GPU | Same as full finetune; batch from VRAM tiers or `THESIS_HRM_FINETUNE_BATCH`. |

---

## Default environment (overridable)

Set by `run_hrm_mlp_finetune_smoke.sh` before invoking the main script:

| Variable | Default | Purpose |
|----------|---------|---------|
| `THESIS_MAX_SAMPLES` | `1000` | Cap training rows (deterministic first N in parquet). |
| `THESIS_CHECKPOINT_ROOT` | `$ROOT/DUMMY/fine-tune/checkpoints/hrm` | Sandbox fine-tune `.safetensors` tree. |
| `THESIS_RESUME_TEMP` | `$ROOT/DUMMY/fine-tune/resume` | Sandbox live resume root. |
| `THESIS_HRM_FINETUNE_LOG` | `DUMMY/fine-tune/logs/hrm_finetune_smoke_${LABELS}label.log` | Tee log path. |
| `THESIS_EPOCHS_FINETUNE` | `3` | Enough epochs to see periodic saves. |
| `THESIS_SAVE_EVERY_STEPS` | `15` | Frequent step-based resume snapshots. |
| `THESIS_SAVE_EVERY_MINUTES` | `1` | Frequent time-based resume snapshots. |
| `THESIS_DETACH` | `none` | Foreground smoke (see training output; production default is `tmux`). |
| `THESIS_NUM_WORKERS` | `4` | DataLoader workers. |
| `THESIS_DATA_ROOT` | *(unset → `$ROOT/data`)* | `--data_root` for training; preflight checks `$THESIS_DATA_ROOT/processed/${STEM}.parquet`. Example: `$ROOT/BACKUP/data` for `BACKUP/data/processed/all-data.parquet`. |
| Label mode | Positional `2` / `3` or `THESIS_HRM_FINETUNE_LABELS` | Same as `run_hrm_finetune_mlp.sh`. |

Inherited from the main launcher / your shell: `CUDA_VISIBLE_DEVICES`, `THESIS_NPROC_PER_NODE`, `THESIS_HRM_FINETUNE_BATCH`, `THESIS_HRM_PRETRAIN_STEM`, etc.

---

## How to run

From **repo root**:

```bash
# 3-class smoke (default 1000 rows)
bash DUMMY/fine-tune/run_hrm_mlp_finetune_smoke.sh

# 2-class smoke (neutral rows dropped; effective rows may be < 1000)
bash DUMMY/fine-tune/run_hrm_mlp_finetune_smoke.sh 2

# Same smoke using an alternate processed tree (e.g. backup copy of all-data.parquet)
THESIS_DATA_ROOT="$ROOT/BACKUP/data" bash DUMMY/fine-tune/run_hrm_mlp_finetune_smoke.sh
```

**Interrupt + resume demo** (recommended single-GPU):

```bash
bash DUMMY/fine-tune/demo_resume_interrupt.sh       # 3-class
bash DUMMY/fine-tune/demo_resume_interrupt.sh 2     # 2-class
# Shorter first phase: FIRST_PHASE_SEC=90 bash DUMMY/fine-tune/demo_resume_interrupt.sh
```

**Multi-GPU:** run the smoke script normally; for interrupt testing, prefer **manual Ctrl+C** then re-run the same smoke command (avoid `timeout` across multiple ranks).

---

## What to verify on the server

1. **Log file** appears under `DUMMY/fine-tune/logs/` and shows training progress.
2. **Periodic resume** artifacts appear under `DUMMY/fine-tune/resume/` during the run.
3. **Final** `*.safetensors` under `DUMMY/fine-tune/checkpoints/hrm/fine-tune/all-data/2-labels/` or `.../3-labels/` (matches label mode).
4. After **demo** phase 1 + phase 2: second run **restores** (watch log / epoch continuation) when `LiveResumeDir` bundle is complete and metadata matches.

---

## Cleanup

```bash
rm -rf DUMMY/fine-tune/logs DUMMY/fine-tune/resume DUMMY/fine-tune/checkpoints
```

---

## Out of scope

- No changes to the original MLP finetune feature plan (`hrm_mlp_head_finetune_44b7207b.plan.md`).
- No dedicated pytest suite for smoke; validation is **operational** (bash + GPU run).
- Smoke does not replace full-dataset finetune; omit `THESIS_MAX_SAMPLES` in production (see main summary doc).
