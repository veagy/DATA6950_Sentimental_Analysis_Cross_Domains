# Thesis training helpers (archived shell scripts)

The former `scripts/thesis/*.sh` and `train_2gpu.ps1` were removed; this document preserves what they did and gives **copy-pastable** replacements from the **repository root** (`ROOT`).

## One-shot pipelines (preferred)

From `ROOT`:

- **HRM encoder-only MLM pretrain ONLY** (no feature encoders, no CNN/RNN/ML queue): `bash scripts/run_hrm_encoder_pretrain_only.sh`. **Dual-GPU SIGTERM resume drill (manual):** read `scripts/smoke_hrm_ddp_sigterm_resume.sh`. — default `CUDA_VISIBLE_DEVICES=0` (single GPU / laptop); set `CUDA_VISIBLE_DEVICES=0,1` on dual-GPU VMs. Output: `checkpoints/pretrain/{stem}/E_HRM1_4Level.safetensors`. **Logs:** `logs/hrm_encoder_pretrain_only.log` (appended; override `THESIS_HRM_PRETRAIN_LOG`). **Detach:** `THESIS_DETACH=tmux|screen|nohup|none`, `THESIS_SESSION` for tmux/screen. Windows (WSL): `.\scripts\run_hrm_encoder_pretrain_only.ps1`
- **Full thesis pretrain pipeline** (HRM MLM as above, **then** five feature-encoder pretrains + CNN/RNN/ML queue): `bash scripts/run_thesis_pretrain.sh`
- **Text finetune (transformers + HRM, frozen backbone, skip `all-data`):** `bash scripts/run_thesis_finetune_text.sh`

Both honor `THESIS_DETACH=tmux|screen|nohup|none`, `THESIS_RESUME_TEMP` (default `logs/resume`), `THESIS_NPROC_PER_NODE`, and `CUDA_VISIBLE_DEVICES`.

### HRM MLM in `run_thesis_pretrain.sh`

Shared encoder JSON: `Code/thesis/config/hrm/E_HRM1_4Level.json`. **MLM pretrain** runs the bare `HierarchicalReasoningModel` (trunk + `lm_head`); artifacts go to **`checkpoints/pretrain/{stem}/E_HRM1_4Level.safetensors`** (no `--n_classes`). **Finetune** uses `HRMClassifierWrapper` with **`--n_classes 2`** or **`3`** → `checkpoints/K-labels/...`. Load MLM weights into finetune with **`--hrm_encoder_ckpt`**.

`run_thesis_finetune_text.sh` runs HRM **twice** per dataset stem (`--n_classes 2` then `3`) so binary and ternary finetune checkpoints are both produced from the same encoder config.

## Purpose

- Repo-local **venv** with CUDA PyTorch
- **CUDA / PyTorch** diagnostics and optional NVIDIA stack recovery
- **Single-GPU** and **dual-GPU** (`torch.distributed.run`) training via `Code/thesis/train/train_single.py`
- **HRM** MLM pretrain on merged corpus: `--pretrain_text_source all_data_parquet` (single `all-data.parquet`) or `all_processed` (all `processed/*.parquet`)
- **CNN/RNN** (and optional ML) **queue** via `Code/thesis/train/train_queue.py` (uses `torchrun`, not a separate shell script)
- **100-D feature encoders (FFNN, CNN, LSTM, RNN, GRU):** canonical pretrain JSONs under `Code/thesis/config/pretrain/2_labels/Pretrain_*.json` train a **denoising autoencoder** (MSE) on `data/transformed/all-data.parquet` with **no** classification head; checkpoints are **encoder-only** at `checkpoints/pretrain/pretrain_{ffnn,cnn,lstm,gru,rnn}.safetensors`. Finetune configs live under `Code/thesis/config/feature_encoder/{2,3}_labels/` and use `train_single.py --encoder_pretrain_ckpt …`. The queue runs the five pretrain jobs first (unless `--skip-feature-encoder-pretrain`), then schedules those finetune jobs together with legacy CNN/RNN configs. Requires **pyarrow** for streaming parquet reads during pretrain.
- **Detached** runners: `nohup`, GNU `screen`, `tmux`

## Conventions

- **`ROOT`**: repository root (all examples assume `cd` there first).
- **`THESIS_PYTHON`**: interpreter to use (default in docs was `python3` or `ROOT/.venv/bin/python` after venv setup).
- **`THESIS_DATA_ROOT`**: passed as `--data_root` where applicable (default often `data`).
- **`CUDA_VISIBLE_DEVICES`**: e.g. `0,1` for two GPUs.
- **`THESIS_NPROC_PER_NODE`**: processes per node for distributed training (default `2`).
- **`THESIS_SKIP_CUDA_PREFLIGHT=1`**: skip GPU checks in the old `train_2gpu.sh` (not recommended).

### Hugging Face / transformer weights

LLM checkpoints live under `checkpoints/deep_learning/llm/<sanitized_model_name>/` (see `Code/models/deep_learning/llm/llm_models.py`). Pre-download with `huggingface-cli` or `huggingface_hub` as described in project docs.

---

## Setup: venv + PyTorch (was `ensure_thesis_venv.sh`)

Idempotent: create `ROOT/.venv`, upgrade pip, install CUDA 12.4 PyTorch wheel + numpy if missing.

```bash
# Skip entirely (use your own interpreter):
#   export THESIS_SKIP_VENV=1

ROOT="$(pwd)"   # from repo root
VENV="$ROOT/.venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124
"$VENV/bin/pip" install -q numpy
export THESIS_PYTHON="$VENV/bin/python"
"$THESIS_PYTHON" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## Diagnostics (was `diagnose_cuda_torch.sh`, `restart_nvidia_stack_sudo.sh`)

```bash
ROOT="$(pwd)"
PY="${THESIS_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

nvidia-smi || true
"$PY" -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.zeros(1, device='cuda:0'))
"

# If nvidia-smi works but PyTorch fails, try (requires sudo):
sudo systemctl restart nvidia-persistenced
# or reboot
```

---

## Core training

### Single process (was `train_one.sh`)

```bash
CONFIG="Code/thesis/config/transformers/2_labels/B3_E_DL1_DistilBERT.json"
STEM="IMDB_Dataset"
DATA_ROOT="${THESIS_DATA_ROOT:-data}"
python3 Code/thesis/train/train_single.py \
  --config "$CONFIG" \
  --dataset_stem "$STEM" \
  --data_root "$DATA_ROOT" \
  ${THESIS_EXTRA_ARGS:-}
```

### Dual GPU (was `train_2gpu.sh`)

The script set `CUDA_VISIBLE_DEVICES`, augmented `LD_LIBRARY_PATH` with venv `nvidia/*/lib` if present, ran a CUDA preflight, then:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PY="${THESIS_PYTHON:-python3}"
NPROC="${THESIS_NPROC_PER_NODE:-2}"

# Optional: match old script — prepend PyTorch-bundled NVIDIA libs
SITEP="$("$PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)" || SITEP=""
NVLIBS=""
if [[ -n "$SITEP" && -d "$SITEP/nvidia" ]]; then
  shopt -s nullglob
  for d in "$SITEP"/nvidia/*/lib; do
    [[ -d "$d" ]] || continue
    NVLIBS="${NVLIBS:+$NVLIBS:}$d"
  done
  shopt -u nullglob
fi
if [[ -n "$NVLIBS" ]]; then
  export LD_LIBRARY_PATH="${NVLIBS}:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" Code/thesis/train/train_single.py "$@"
```

Pass through any `train_single.py` flags after the final `"$@"`.

### Detached dual-GPU (was `train_2gpu_nohup.sh`, `train_screen.sh`, `train_tmux.sh`)

Use the same `LD_LIBRARY_PATH` / `THESIS_PYTHON` / `CUDA_VISIBLE_DEVICES` setup as in **Dual GPU** above, then:

```bash
ROOT="$(pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PY="${THESIS_PYTHON:-python3}"
NPROC="${THESIS_NPROC_PER_NODE:-2}"
mkdir -p "$ROOT/logs"

# Fill in training flags (same trailing args the old train_2gpu.sh would pass):
ARGS=( --config Code/thesis/config/cnn/2_labels/B9_CNN_Text.json --dataset_stem IMDB_Dataset )

nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
  Code/thesis/train/train_single.py "${ARGS[@]}" \
  >>"$ROOT/logs/nohup_train.log" 2>&1 &
echo "PID $! — tail -f $ROOT/logs/nohup_train.log"
```

**screen** (session name `NAME`, default `thesis_train`): start a detached shell in repo root, then run the same `torch.distributed.run` line as above; the old script ended with `exec bash` to keep the session open.

**tmux** (session `SESSION`, default `thesis_train`): same pattern as screen. Quote the inner command carefully so arguments with spaces are not split.

---

## Preflight smoke (was `preflight_smoke_train.sh`)

Small-sample checks writing only to `checkpoints/.preflight_smoke` and `logs/.preflight_smoke`. Optional: `SKIP_HRM=1`, `SKIP_CNN=1`, `SKIP_ML=1`.

```bash
ROOT="$(pwd)"
cd "$ROOT"
STEM="${1:-IMDB_Dataset}"
DATA_ROOT="${PREFLIGHT_DATA_ROOT:-$ROOT/data}"
CK="$ROOT/checkpoints/.preflight_smoke"
LG="$ROOT/logs/.preflight_smoke"
mkdir -p "$CK" "$LG"
export MAX_SAMPLES="${MAX_SAMPLES:-8000}"
export EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-1}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GC_EVERY="${GC_EVERY:-25}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

if [[ "${SKIP_HRM:-0}" != "1" ]]; then
  PY="${THESIS_PYTHON:-python3}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  NPROC="${THESIS_NPROC_PER_NODE:-2}"
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" Code/thesis/train/train_single.py \
    --config "$ROOT/Code/thesis/config/hrm/E_HRM1_4Level.json" \
    --n_classes 2 \
    --dataset_stem "$STEM" \
    --data_root "$DATA_ROOT" \
    --checkpoint_root "$CK" \
    --log_dir "$LG" \
    --phase pretrain \
    --pretrain_text_source all_processed \
    --epochs_pretrain "$EPOCHS_PRETRAIN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --gc_every "$GC_EVERY" \
    --max_samples "$MAX_SAMPLES" \
    --lr 2e-5
fi

if [[ "${SKIP_CNN:-0}" != "1" ]]; then
  PY="${THESIS_PYTHON:-python3}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  NPROC="${THESIS_NPROC_PER_NODE:-2}"
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" Code/thesis/train/train_single.py \
    --config "$ROOT/Code/thesis/config/cnn/2_labels/B9_CNN_Text.json" \
    --dataset_stem "$STEM" \
    --data_root "$DATA_ROOT" \
    --checkpoint_root "$CK" \
    --log_dir "$LG" \
    --phase finetune \
    --epochs_finetune 1 \
    --epochs_pretrain 1 \
    --batch_size 16 \
    --num_workers 2 \
    --max_samples 512
fi

if [[ "${SKIP_ML:-0}" != "1" ]]; then
  python3 "$ROOT/Code/thesis/train/train_single.py" \
    --config "$ROOT/Code/thesis/config/ml/2_labels/E_ML1_LogisticRegression.json" \
    --dataset_stem "$STEM" \
    --data_root "$DATA_ROOT" \
    --checkpoint_root "$CK" \
    --log_dir "$LG" \
    --phase finetune \
    --epochs_finetune 1 \
    --batch_size 8 \
    --num_workers 0 \
    --max_samples 500
fi

# rm -rf "$CK" "$LG"
```

---

## Rebuild merged `all-data` (processed + transformed)

### Processed merge

`data/processed/all-data.parquet` is the **concatenation** of every `data/processed/*.parquet` except `all-data`, with unified columns `text`, `sentiment_value`, and optional `source_stem` (use `source_stem` to split back into per-stem files). Row count must equal the sum of source file rows; the script aborts otherwise.

```bash
ROOT="$(pwd)"
PY="${THESIS_PYTHON:-$ROOT/.venv/bin/python}"
"$PY" Code/thesis/data/merge_all_data_parquet.py
```

After you have verified the merge, you may delete the per-stem processed parquets (they can be recreated by grouping on `source_stem`):

```bash
"$PY" Code/thesis/data/merge_all_data_parquet.py --delete-sources-after --confirm-delete-sources
```

### Transformed merge (optional)

`Code/thesis/data/merge_all_transformed_parquet.py` builds `data/transformed/all-data.parquet` from every `data/transformed/*.parquet` except `all-data`, appending `source_stem`. It enforces **identical column names** across shards (same order as the first file) and the same row-sum check as the processed merger. Optional cleanup:

```bash
"$PY" Code/thesis/data/merge_all_transformed_parquet.py
"$PY" Code/thesis/data/merge_all_transformed_parquet.py --delete-sources-after --confirm-delete-sources
```

**UMAP caveat:** Each stem is normally embedded with its own UMAP fit in `embed_reduce.py`. Concatenating existing transformed shards preserves rows and labels but **mixes embedding spaces** (100D coordinates are not guaranteed comparable across stems). For **one** consistent feature space over the full corpus, prefer merging **processed** first, then:

```bash
"$PY" Code/thesis/data/embed_reduce.py --only all-data --force
```

`embed_reduce.py` loads MiniLM on **CUDA**; full-corpus embedding can take a long time.

### Queue and per-stem training after cleanup

If you delete every per-stem parquet and only `all-data.parquet` remains under `data/transformed/`, `Code/thesis/train/train_queue.py` will list **no** stems unless you set **`THESIS_QUEUE_INCLUDE_ALL_DATA=1`** (truthy: `1`, `true`, `yes`), because it **omits** `all-data` by default. The same applies to `train_single.py --dataset_stem` when it expects `data/transformed/{stem}.parquet`: restore stems by splitting `all-data.parquet` on `source_stem`, or set the env var above to train on the merged feature file as `all-data`.

**CNN/RNN queue (when multiple stems exist):** `train_queue.py` **omits** the `all-data` stem by default so jobs are not scheduled on the entire merged feature set in addition to each dataset stem. Set **`THESIS_QUEUE_INCLUDE_ALL_DATA=1`** to include `all-data` in the queue.

---

## HRM merged pretrain

### Single-GPU nohup (was `hrm_pretrain_merged_nohup.sh`)

`python3 Code/thesis/train/train_single.py` with `--phase pretrain --pretrain_text_source all_processed`, config `Code/thesis/config/hrm/E_HRM1_4Level.json` plus `--n_classes 2` or `3`, logs under `logs/hrm_pretrain_merged_*`.

### Dual-GPU nohup (was `hrm_pretrain_merged_2gpu_nohup.sh`)

Same args via the dual-GPU `torch.distributed.run` wrapper; defaults included `EPOCHS_PRETRAIN=8`, `BATCH_SIZE=8`, `NUM_WORKERS=0`, `GC_EVERY=50`, optional `MAX_SAMPLES`.

### Sequential 2-label then 3-label (was `hrm_pretrain_merged_2gpu_both_labels.sh`)

Run the dual-GPU pretrain twice with the **same** `Code/thesis/config/hrm/E_HRM1_4Level.json`, first `--n_classes 2`, then `--n_classes 3`. Checkpoints under `checkpoints/2-labels/<stem>/` and `checkpoints/3-labels/<stem>/`. Supported `THESIS_RESUME_TEMP` as `--resume_temp_root` when set.

### Detached both-labels (was `hrm_pretrain_merged_2gpu_both_labels_nohup.sh`)

`nohup` wrapping the sequential script; PID file `logs/hrm_pretrain_merged_both.pid`, log `logs/hrm_pretrain_merged_2gpu_2l_then_3l_<ts>.log`. `THESIS_HRM_LOG_TS` could align timestamps with the pipeline script.

---

## CNN/RNN queue (was `queue_cnn_rnn_2gpu_after_wait.sh`)

Runs `Code/thesis/train/train_queue.py` with optional waits and stability window on `data/transformed/*.parquet`.

```bash
ROOT="$(pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"

extra=()
for p in ${WAIT_PID:-}; do
  [[ -n "$p" ]] && extra+=( --wait-pid "$p" )
done
[[ -n "${WAIT_PID_FILE:-}" && -f "${WAIT_PID_FILE}" ]] && extra+=( --wait-pid-file "${WAIT_PID_FILE}" )
[[ "${SKIP_STABLE:-0}" != "1" ]] && extra+=( --stable-seconds "${STABLE_SECONDS:-120}" )
[[ -n "${THESIS_DATA_ROOT:-}" ]] && extra+=( --data_root "${THESIS_DATA_ROOT}" )

exec "$PY" Code/thesis/train/train_queue.py "${extra[@]}" "$@"
```

**Detached:** same command under `nohup` → `logs/queue_cnn_rnn_nohup.log` (or `QUEUE_NOHUP_LOG`), or via `screen`/`tmux` session names `thesis_queue_cnn_rnn` / `NAME` / `SESSION` overrides.

**Classical ML in queue:** add `--include-ml` (former `INCLUDE_ML=1` in the pipeline script).

---

## End-to-end pipeline (was `hrm_then_feature_queue_both_nohup.sh`)

1. Optionally run `ensure_thesis_venv` equivalent; set `THESIS_PYTHON`, `LD_LIBRARY_PATH` like `train_2gpu.sh`.
2. CUDA preflight (unless `THESIS_ALLOW_START_WITHOUT_CUDA=1`).
3. Start `hrm_pretrain_merged_2gpu_both_labels_nohup` equivalent (writes `logs/hrm_pretrain_merged_both.pid`).
4. Start queue with `WAIT_PID_FILE=$ROOT/logs/hrm_pretrain_merged_both.pid` and optional `--include-ml`.
5. Write `logs/pipeline_metadata.json` (run id, paths, resume hints).

**Resume queue after interrupt** (from old metadata): set `THESIS_QUEUE_RESUME_DIR=logs/queue_cnn_rnn_<ts>`, `SKIP_STABLE=1`, then run `queue_cnn_rnn_2gpu_after_wait.sh` equivalent with `--skip-wait` (see `docs/README.md`).

---

## Windows (was `train_2gpu.ps1`)

PowerShell from repo root: set `CUDA_VISIBLE_DEVICES` (default `0`), run `python Code/thesis/train/train_single.py` with **caller-supplied** arguments (`@args`). There is no built-in `torch.distributed.run` wrapper in that file; use the bash dual-GPU flow on Linux/WSL or invoke `torch.distributed.run` manually in PowerShell.

---

## Quick reference table

| Former script | Role |
|---------------|------|
| `ensure_thesis_venv.sh` | `.venv` + CUDA PyTorch |
| `diagnose_cuda_torch.sh` | Driver + torch sanity |
| `restart_nvidia_stack_sudo.sh` | `sudo systemctl restart nvidia-persistenced` |
| `train_one.sh` | `train_single.py` one GPU / CPU |
| `train_2gpu.sh` | `torch.distributed.run` × 2 |
| `train_2gpu_nohup.sh` / `train_screen.sh` / `train_tmux.sh` | Detached `train_2gpu` |
| `preflight_smoke_train.sh` | Small HRM + CNN + ML smoke |
| `hrm_pretrain_merged_*.sh` | HRM MLM on merged processed parquet |
| `queue_cnn_rnn_2gpu_after_wait.sh` | `train_queue.py` |
| `queue_cnn_rnn_2gpu_*` | Detached queue |
| `hrm_then_feature_queue_both_nohup.sh` | HRM then queue + metadata |
| `train_2gpu.ps1` | Windows single-process helper |

For **resume behavior**, **`THESIS_RESUME_TEMP`**, and **queue resume**, see [docs/README.md](../docs/README.md).
