# HRM MLP head fine-tune — implementation summary

**Last updated:** 2026-03-31  

**Plan reference (read-only):** `c:\Users\ravul\.cursor\plans\hrm_mlp_head_finetune_44b7207b.plan.md`

---

## Purpose

Frozen **HRM encoder** (`HierarchicalReasoningModel`) plus a **trainable deep MLP head** for **2- or 3-way** sentiment classification; **supervised** fine-tune on **processed** parquet; compatible with **DDP** and **periodic resume** checkpoints (same family of knobs as HRM encoder pretrain).

---

## Overview

- **Encoder:** `HierarchicalReasoningModel` weights are loaded from MLM pretrain (e.g. `E_HRM1_4Level.safetensors`) and **frozen** during supervised fine-tune.
- **Head:** A fixed-architecture MLP maps the **100-dimensional** mean-pooled embedding (`output_embed_dim`) to **2 or 3** logits. Training uses **`CrossEntropyLoss` on logits**; there is **no `Softmax` layer inside the module** (apply `softmax` at inference if you need probabilities).
- **Data:** `data/processed/{dataset_stem}.parquet` with text + labels (`sentiment_value` / `sentiment` inferred). **3-label:** all classes. **2-label:** rows with neutral label (`2`) are **dropped by default** for HRM 2-class runs.
- **Distribution:** Launcher uses `python -m torch.distributed.run --nproc_per_node=…` consistent with HRM encoder pretrain scripts.

---

## Architecture

```mermaid
flowchart LR
  text[text_batch] --> enc[HRM_encoder_frozen]
  enc --> pool[mean_pool_100d]
  pool --> mlp[MLP_head_trainable]
  mlp --> logits[K_logits]
  logits --> ce[CrossEntropyLoss]
```

**MLP topology** (implemented in `build_sentiment_mlp_head`):  
100 → Linear → 320 → ReLU → 640 → GELU → 1250 → GELU → 640 → ReLU → 320 → GELU → `K` (2 or 3).

---

## Data

| Item | Detail |
|------|--------|
| Default parquet | `data/processed/all-data.parquet` when `--dataset_stem all-data` |
| Text / labels | `ParquetTextDataset` infers text column and label column (`sentiment_value`, `sentiment`, or similar via `_infer_label_column`) |
| Class indices | From `coerce_label_int`: negative `0`, positive `1`, neutral `2` |
| 3-label finetune | All rows with labels 0 / 1 / 2 |
| 2-label finetune | Rows with label `2` (neutral) **excluded by default** for HRM + `n_classes == 2`; use `--no_hrm_exclude_neutral` on `train_single.py` to keep them |

---

## Encoder and output checkpoints

| Artifact | Path pattern |
|----------|----------------|
| Encoder (load) | `checkpoints/hrm/pretrain/{stem}/E_HRM1_4Level.safetensors` — `{stem}` default `all-data`; override with `THESIS_HRM_ENCODER_CKPT` or `--hrm_encoder_ckpt` |
| Fine-tune (save) | With `--hrm_finetune_checkpoint_layout` and `--checkpoint_root .../checkpoints/hrm`: `checkpoints/hrm/fine-tune/{dataset_stem}/{K}-labels/{config_stem}.safetensors` |
| Example filenames | `E_HRM1_4Level_ft_mlp_2label.safetensors`, `E_HRM1_4Level_ft_mlp_3label.safetensors` |

---

## Logs and resume

| Artifact | Location |
|----------|----------|
| Tee logs | `logs/hrm_finetune_mlp_2label.log`, `logs/hrm_finetune_mlp_3label.log` (override `THESIS_HRM_FINETUNE_LOG`) |
| Live resume | `logs/resume/` by default (`THESIS_RESUME_TEMP`) |
| Layout flag from env | `THESIS_HRM_FINETUNE_CKPT_LAYOUT=1|true|yes` sets the same behavior as `--hrm_finetune_checkpoint_layout` in `train_single.py` |

---

## Implemented components vs plan

| Item | Status |
|------|--------|
| `build_sentiment_mlp_head` + `HRMClassifierWrapper(..., head=None)` optional custom head | Done |
| JSON configs with `classification_head.type = mlp_sentiment_v1` and `num_classes` 2 / 3 | Done |
| `model_factory` pops `classification_head`, builds MLP, wraps encoder | Done |
| `_resolve_n_classes` reads `classification_head.num_classes` | Done |
| `ParquetTextDataset(..., exclude_neutral=True)` drops label `2` | Done |
| `train_single`: HRM 2-class + default exclude neutral; `--no_hrm_exclude_neutral` to keep neutrals | Done |
| `train_single`: `--hrm_finetune_checkpoint_layout`, env `THESIS_HRM_FINETUNE_CKPT_LAYOUT` | Done |
| Scripts: VRAM-tier batch, `THESIS_HRM_FINETUNE_BATCH` / `THESIS_HRM_BATCH`, dual-GPU `nproc` from `CUDA_VISIBLE_DEVICES` | Done |
| Scripts: **all rows** unless `THESIS_MAX_SAMPLES` set | Done |
| Scripts: default **`THESIS_DETACH=tmux`**; `THESIS_DETACH=none` for foreground | Done |

---

## File and folder locations

### Configs (thesis)

| Path | Role |
|------|------|
| `Code/thesis/config/hrm/E_HRM1_4Level.json` | Base encoder spec (MLM pretrain reference; same inner `config` as MLP finetune configs) |
| `Code/thesis/config/hrm/E_HRM1_4Level_ft_mlp_2label.json` | Encoder `config` + `classification_head` for **2** classes |
| `Code/thesis/config/hrm/E_HRM1_4Level_ft_mlp_3label.json` | Encoder `config` + `classification_head` for **3** classes |

### Code

| Path | Role |
|------|------|
| `Code/models/deep_learning/hrm/hrm_model.py` | `build_sentiment_mlp_head`, `HierarchicalReasoningModel`, `HRMClassifierWrapper` |
| `Code/models/deep_learning/hrm/__init__.py` | Exports including `build_sentiment_mlp_head` |
| `Code/thesis/common/model_factory.py` | Builds encoder + MLP head from JSON |
| `Code/thesis/common/datasets.py` | `ParquetTextDataset`, `coerce_label_int`, `exclude_neutral` |
| `Code/thesis/train/train_single.py` | CLI, HRM finetune loop, checkpoint paths, dataset wiring |

### Scripts

| Path | Role |
|------|------|
| `scripts/run_hrm_finetune_mlp.sh` | Main launcher: labels `2`/`3`, DDP, logs, tmux default |
| `scripts/run_hrm_finetune_mlp_2label.sh` | Wrapper → `run_hrm_finetune_mlp.sh 2` |
| `scripts/run_hrm_finetune_mlp_3label.sh` | Wrapper → `run_hrm_finetune_mlp.sh 3` |
| `scripts/run_hrm_finetune_mlp.ps1` | Windows → WSL `bash scripts/run_hrm_finetune_mlp.sh` |

### Runtime I/O (quick index)

See sections **Data**, **Encoder and output checkpoints**, and **Logs and resume** above for paths and overrides. Optional GPU sampling log: `logs/hrm_finetune_gpu_monitor.log`.

### Related documentation

| Path | Notes |
|------|------|
| `docs/train-optimize.txt` | Throughput tips (batch, DDP, workers, TF32) |
| `docs/hrm_encoder_pretrain_runbook.md` | Encoder MLM pretrain; finetune builds on that checkpoint |
| `scripts/run_hrm_encoder_pretrain_only.sh` | Pretrain launcher pattern mirrored by finetune script |

---

## How to use

### Recommended (repo root, Linux / WSL)

```bash
export CUDA_VISIBLE_DEVICES=0,1
# Optional: fixed per-GPU batch (else VRAM tier picks 2/8/24/64)
# export THESIS_HRM_FINETUNE_BATCH=64

bash scripts/run_hrm_finetune_mlp_3label.sh   # then 2-label
bash scripts/run_hrm_finetune_mlp_2label.sh
```

- **Detach:** default is **tmux** (`THESIS_DETACH=tmux`). Attach: `tmux attach -t hrm_finetune_mlp_3label` (session name includes label mode).
- **Foreground:** `THESIS_DETACH=none bash scripts/run_hrm_finetune_mlp.sh 3`
- **Full dataset:** do **not** set `THESIS_MAX_SAMPLES`. Set it only for smoke tests.

### Windows (PowerShell)

From repo root:

```powershell
.\scripts\run_hrm_finetune_mlp.ps1      # default 3-class via WSL
.\scripts\run_hrm_finetune_mlp.ps1 2    # 2-class
```

Set environment variables in WSL before calling if needed.

### Manual `torchrun` (illustrative)

```bash
python -m torch.distributed.run --nproc_per_node=2 \
  Code/thesis/train/train_single.py \
  --config Code/thesis/config/hrm/E_HRM1_4Level_ft_mlp_3label.json \
  --dataset_stem all-data \
  --data_root data \
  --checkpoint_root checkpoints/hrm \
  --log_dir logs \
  --phase finetune \
  --epochs_pretrain 0 \
  --epochs_finetune 3 \
  --batch_size 32 \
  --hrm_encoder_ckpt checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors \
  --hrm_finetune_checkpoint_layout \
  --num_workers 8 \
  --gc_every 0 \
  --save_every_minutes 5 \
  --save_every_steps 1000
```

Add `--no_hrm_exclude_neutral` only if you need **2-class** training **without** dropping neutral rows.

### Label semantics

- `coerce_label_int`: negative → `0`, positive → `1`, neutral → `2`.
- **2-class HRM finetune:** neutral rows skipped unless `--no_hrm_exclude_neutral`.

### Batch size and hardware

| Profile | VRAM (guide) | Suggested first try (`THESIS_HRM_FINETUNE_BATCH` per GPU) |
|---------|----------------|------------------------------------------------------------|
| 2× RTX Pro 6000 Blackwell | 96 GB | 32–64 (tiers use **64** if ≥80 GB detected) |
| 2× NVIDIA L40S | 48 GB | 8–24 (tier **24** if 24–80 GB, **8** if 8–24 GB) |

Script tier defaults (if neither `THESIS_HRM_FINETUNE_BATCH` nor `THESIS_HRM_BATCH` is set): **&lt;8 GB → 2**, **&lt;24 GB → 8**, **&lt;80 GB → 24**, **else → 64**.

---

## Smoke test (sandbox)

Dedicated write-up for the smoke scripts and verification checklist: [`docs/summary/hrm_mlp_finetune_smoke_implementation.md`](hrm_mlp_finetune_smoke_implementation.md).

**Purpose:** On an Ubuntu + NVIDIA box, validate DDP (or 1-GPU) MLP finetune with **~1000 rows** from `data/processed/all-data.parquet` (or `THESIS_HRM_PRETRAIN_STEM`) without writing production checkpoints under `checkpoints/hrm/`. Logs, live resume bundles, and fine-tune outputs stay under **`DUMMY/fine-tune/`**.

**Encoder:** The smoke script sets `THESIS_CHECKPOINT_ROOT` to the sandbox but leaves the encoder at the **real** pretrain path `checkpoints/hrm/pretrain/{stem}/E_HRM1_4Level.safetensors` unless you already set `THESIS_HRM_ENCODER_CKPT`.

**Commands (repo root):**

```bash
bash DUMMY/fine-tune/run_hrm_mlp_finetune_smoke.sh      # 3-class, 1000 rows
bash DUMMY/fine-tune/run_hrm_mlp_finetune_smoke.sh 2    # 2-class
# or: THESIS_HRM_FINETUNE_LABELS=2 bash DUMMY/fine-tune/run_hrm_mlp_finetune_smoke.sh
```

**Interrupt / resume demo (single-GPU default):** Phase 1 sends **SIGINT** after 120s via `timeout`, then phase 2 runs the same smoke command so `LiveResumeDir` can restore if the bundle is complete.

```bash
bash DUMMY/fine-tune/demo_resume_interrupt.sh     # 3-class
bash DUMMY/fine-tune/demo_resume_interrupt.sh 2   # 2-class
```

For **multi-GPU**, prefer manual **Ctrl+C** once, then re-run the smoke script; `timeout` across `torchrun` ranks is easier to get wrong.

**Useful env overrides:** `THESIS_MAX_SAMPLES`, `THESIS_SAVE_EVERY_STEPS`, `THESIS_SAVE_EVERY_MINUTES`, `THESIS_EPOCHS_FINETUNE`, `CUDA_VISIBLE_DEVICES`, `THESIS_HRM_FINETUNE_BATCH`, `THESIS_NUM_WORKERS`. Defaults include `THESIS_DETACH=none`, `THESIS_SAVE_EVERY_STEPS=15`, `THESIS_SAVE_EVERY_MINUTES=1` so periodic resume saves fire quickly.

**Where artifacts go:**

| Kind | Path |
|------|------|
| Tee log | `DUMMY/fine-tune/logs/hrm_finetune_smoke_2label.log` or `..._3label.log` |
| Live resume | `DUMMY/fine-tune/resume/` (slugged subdirs from `train_single`) |
| Fine-tune checkpoints | `DUMMY/fine-tune/checkpoints/hrm/fine-tune/all-data/{K}-labels/*.safetensors` (with default stem and layout flag from the launcher) |

**Cleanup:**

```bash
rm -rf DUMMY/fine-tune/logs DUMMY/fine-tune/resume DUMMY/fine-tune/checkpoints
```

**Note:** **2-label** smoke uses **at most** `THESIS_MAX_SAMPLES` rows from parquet, but **fewer** if many rows are neutral (dropped for 2-class HRM by default).

---

## Environment variables (quick reference)

| Variable | Purpose |
|----------|---------|
| `THESIS_PYTHON` | Python binary (default `python3`) |
| `THESIS_DETACH` | `tmux` (default) \| `none` \| `screen` \| `nohup` |
| `THESIS_SESSION` | tmux/screen session name prefix |
| `CUDA_VISIBLE_DEVICES` | GPU list; `nproc` defaults to count of indices |
| `THESIS_NPROC_PER_NODE` | Override process count |
| `THESIS_HRM_FINETUNE_BATCH` | Per-GPU `--batch_size` (highest priority) |
| `THESIS_HRM_BATCH` | Fallback batch if finetune batch unset |
| `THESIS_CHECKPOINT_ROOT` | Default `checkpoints/hrm` |
| `THESIS_HRM_PRETRAIN_STEM` | Parquet / pretrain subfolder stem (default `all-data`) |
| `THESIS_HRM_ENCODER_CKPT` | Path to encoder `.safetensors` |
| `THESIS_EPOCHS_FINETUNE` | Default `3` in script |
| `THESIS_MAX_SAMPLES` | Cap rows (smoke only) |
| `THESIS_NUM_WORKERS`, `THESIS_GC_EVERY` | DataLoader / GC |
| `THESIS_SAVE_EVERY_MINUTES`, `THESIS_SAVE_EVERY_STEPS`, `THESIS_MIN_SAVE_INTERVAL_SEC` | Periodic resume saves |
| `THESIS_RESUME_TEMP` | Resume root (default `logs/resume`) |
| `THESIS_AMP_BF16`, `THESIS_DATALOADER_PERSISTENT` | Passed through when enabled |
| `THESIS_HRM_FINETUNE_LOG` | Override tee log path |
| `THESIS_HRM_FINETUNE_LABELS` | `2` or `3` if not passing positional arg |
| `THESIS_HRM_FINETUNE_CKPT_LAYOUT` | `1`/`true`/`yes` → same as `--hrm_finetune_checkpoint_layout` in `train_single` |

---

## Out of scope / notes

- **No softmax in-module** during training; use `CrossEntropyLoss` with raw logits.
- **J-Star** review/audit is not part of this feature; run separately if desired.
- **`phase all`** rebuilding the wrapper after MLM in the same run does not re-apply the MLP head from JSON; use **finetune-only** runs with the MLP configs for the intended head.
- Encoder **must** exist at `THESIS_HRM_ENCODER_CKPT` (or default path); the script prints a warning if the file is missing.
