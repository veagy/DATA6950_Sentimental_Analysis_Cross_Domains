# HRM encoder MLM pretrain — runbook

Encoder-only masked language modeling on **`data/processed/all-data.parquet`**. Entry script: [`scripts/run_hrm_encoder_pretrain_only.sh`](../scripts/run_hrm_encoder_pretrain_only.sh).

## Data

- **File:** `data/processed/all-data.parquet` (repo root = directory that contains `scripts/` and `data/`).
- **Build:** [`Code/thesis/data/merge_all_data_parquet.py`](../Code/thesis/data/merge_all_data_parquet.py) — see [`data/processed/README.md`](../data/processed/README.md).

After pretrain, you can sanity-check **encoder-only** mean-pooled embeddings (**shape `[B, output_embed_dim]`**, 100 for `E_HRM1_4Level`) by loading the final safetensors with [`Code/thesis/tools/hrm_embed_smoke.py`](../Code/thesis/tools/hrm_embed_smoke.py). The `--parquet` path reads the **first** `--parquet-rows` rows from the `text` column (not a random sample). From repo root (PowerShell):

```powershell
python Code/thesis/tools/hrm_embed_smoke.py `
  --checkpoint checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors `
  --parquet data/processed/all-data.parquet `
  --parquet-rows 8 `
  --device cpu
```

Omit `--device cpu` for CUDA when available; add `--print-json` for full vectors. Requires `pandas` and `pyarrow` for parquet reads.

## Default run config (script)

- **Epochs:** `THESIS_EPOCHS_PRETRAIN` defaults to **2** when unset.
- **Batch:** VRAM-based defaults (2 / 8 / 24 / 64); override with **`THESIS_HRM_BATCH`** (e.g. `64` on cloud, `1` on 8 GB).
- **Optional smoke cap:** `THESIS_MAX_SAMPLES` → `--max_samples`.

## Checkpoints and periodic resume

| Concern | Location |
|--------|----------|
| Step/time triggers + min interval | [`train_single.py`](../Code/thesis/train/train_single.py) — `_maybe_periodic_resume_save`; CLI `--save_every_steps`, `--save_every_minutes`, `--min_save_interval_sec` |
| Resume bundle (model + optimizer + scaler + meta) | [`resume_checkpoint.py`](../Code/thesis/common/resume_checkpoint.py) — `LiveResumeDir`, `ResumeMeta.steps_completed_in_epoch` |
| Restore + batch skip | `try_restore_training` → `(epoch_next, skip_batches)`; [`train_loop_hrm_mlm`](../Code/thesis/train/train_single.py) skips batches on resumed epoch |
| SIGINT/SIGTERM | `InterruptSave` + save in `train_loop_hrm_mlm` |

Live bundles live under **`logs/resume/`** (or `THESIS_RESUME_TEMP`). Final deliverable weights: **`checkpoints/pretrain/{stem}/E_HRM1_4Level.safetensors`** after a full successful pretrain (live dir is cleared).

## Forward pass and backpropagation (MLM)

- **Forward:** `_hrm_mlm_step` — tokenize, ~15% random mask, `model(masked, pretrain=True)`, cross-entropy with `ignore_index=-100`.
- **Backward:** `train_loop_hrm_mlm` — AMP `autocast`, `GradScaler.scale(loss).backward()`, `scaler.step(opt)`, `scaler.update()`.

## Distributed computing (DDP)

- **Launch:** `python -m torch.distributed.run --nproc_per_node=…` in the shell script.
- **Init / device:** [`distributed.py`](../Code/thesis/common/distributed.py) — `init_distributed_from_env`, NCCL, `torch.cuda.set_device(local_rank)`.
- **Model:** `wrap_ddp` with `find_unused_parameters=True` for HRM MLM (some params unused in a given forward).
- **Data:** `DistributedSampler` + `set_sampler_epoch` + per-rank `Generator` / optional `worker_init_fn` in `_text_loader`.

## Related plans

- [`.cursor/plans/hrm_pretrain_gpu_validation_3e7152b9.plan.md`](../.cursor/plans/hrm_pretrain_gpu_validation_3e7152b9.plan.md)
- [`docs/plans/pretrain_scripts_blackwell_resume_3c0bbcf9.plan.md`](plans/pretrain_scripts_blackwell_resume_3c0bbcf9.plan.md)

## Tests

- Periodic saves + final ckpt: `Code/test/test_hrm_encoder_pretrain_integration_local.py::test_hrm_encoder_pretrain_periodic_saves_and_final_ckpt`
- Resume after interrupt: `test_hrm_encoder_pretrain_resume_after_sigterm`
