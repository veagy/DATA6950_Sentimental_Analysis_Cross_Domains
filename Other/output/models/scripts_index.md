# Shell scripts under `scripts/`

## `moe_ddp_launch.sh`

    # shellcheck shell=bash
    # Source from repo scripts after setting MOE_REPO_ROOT to the TEMP checkout root.
    # torchrun is often missing from PATH; pip/venv installs expose the same via python -m.
    moe_ddp_launch() {
      local nproc=$1
      shift
      if command -v torchrun >/dev/null 2>&1; then
        torchrun --nproc_per_node="$nproc" "$@"
      elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/torchrun" ]]; then
        "${VIRTUAL_ENV}/bin/torchrun" --nproc_per_node="$nproc" "$@"
      elif [[ -n "${MOE_REPO_ROOT:-}" && -x "${MOE_REPO_ROOT}/.venv/bin/torchrun" ]]; then
        "${MOE_REPO_ROOT}/.venv/bin/torchrun" --nproc_per_node="$nproc" "$@"
      else
        # Single-node multi-GPU: --standalone avoids rdzv setup; same role as torchrun here.
        python3 -m torch.distributed.run --standalone --nproc_per_node="$nproc" "$@"
      fi
    }

## `prepare_moe_dummy_parquet.sh`

    #!/usr/bin/env bash
    # Slice aligned rows into TEMP/DUMMY/data/{processed,transformed} for MoE smoke.
    # Usage:
    #   bash scripts/prepare_moe_dummy_parquet.sh
    #   DATA_ROOT=/path/to/data SOURCE_STEM=all-data N_ROWS=1000 bash scripts/prepare_moe_dummy_parquet.sh
    #   SYNTHETIC=1 bash scripts/prepare_moe_dummy_parquet.sh   # no source parquets
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$ROOT"
    export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
    ARGS=(--repo-root "$ROOT" --n-rows "${N_ROWS:-1000}" --out-stem "${OUT_STEM:-moe_dummy_1k}")
    if [[ -n "${SOURCE_STEM:-}" ]]; then
      ARGS+=(--source-stem "$SOURCE_STEM")
    fi
    if [[ -n "${DATA_ROOT:-}" ]]; then
      ARGS+=(--data-root "$DATA_ROOT")
    fi
    if [[ "${SYNTHETIC:-0}" == "1" ]]; then
      ARGS+=(--synthetic)
    fi

## `queue_transformer_mlp_all8_after_hrm_2label.sh`

    #!/usr/bin/env bash
    # Wait until the current HRM 2-label MLP finetune (train_single workers on
    # E_HRM1_4Level_ft_mlp_2label.json) has fully exited, then start the eight
    # transformer MLP finetunes in a new detached tmux session.
    #
    # Does not signal or attach to the running HRM job — polling only.
    #
    # Env:
    #   THESIS_PYTHON — passed into the transformer run (default: $ROOT/.venv/bin/python)
    #   THESIS_QUEUE_POLL_SEC — sleep between checks (default: 60)
    #   THESIS_QUEUE_AFTER_PGPATTERN — extended regex for pgrep -f (default: HRM 2-label finetune)
    #   THESIS_QUEUE_TMUX_SESSION — tmux session for transformer run (default: transformer_ft_all8)
    #   Extra env vars are forwarded into the tmux command via inline export (set before launching this script).
    #
    # Typical use (from anywhere):
    #   nohup bash /path/to/TEMP/scripts/queue_transformer_mlp_all8_after_hrm_2label.sh \
    #     >> /path/to/TEMP/logs/queue_transformer_after_hrm_2l.log 2>&1 &
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

## `run_b11_cnn_lstm_stack_gelu_3gpu.sh`

    #!/usr/bin/env bash
    # B11: frozen CNN→LSTM pretrain stack + GeLU head (DDP on 3 GPUs).
    #
    # WAIT until these finish first (disk + GPU contention):
    #   - logs/mlp_geLU_head_ddp_ALL_CONFIGS_*.log
    #   - logs/ml_bc_queue_rerun_*.log
    # Then run manually, e.g.:
    #   CUDA_VISIBLE_DEVICES=1,2,3 /path/to/run_b11_cnn_lstm_stack_gelu_3gpu.sh
    #
    set -euo pipefail
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
    PORT="${MASTER_PORT:-29620}"
    
    run_one() {
      local CFG="$1"
      echo "=== B11 stack: $CFG (port $PORT) ==="
      "${ROOT}/.venv/bin/torchrun" --nproc_per_node=3 --master_port="${PORT}" \
        "${ROOT}/Code/thesis/train/train_b11_cnn_lstm_stack_gelu_ddp.py" \

## `run_dataset_analysis.sh`

    #!/usr/bin/env bash
    # Dataset EDA for Parquet files under data/ → output/dataset analysis/
    #
    # From TEMP (repo) root:
    #   bash scripts/run_dataset_analysis.sh
    #
    # Environment (optional):
    #   THESIS_PYTHON — default python3
    #   THESIS_DATA_ROOT, THESIS_OUTPUT_DIR — passed as --data-root / --output-dir
    #
    # Flags after -- are forwarded to analyze_thesis_datasets.py, e.g.:
    #   bash scripts/run_dataset_analysis.sh -- --max-rows-per-file 10000 --plots
    
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$ROOT"
    

## `run_export_model_documentation.sh`

    #!/usr/bin/env bash
    # Export model documentation catalog → output/models/
    #
    # From TEMP (repo) root:
    #   bash scripts/run_export_model_documentation.sh
    #
    # Environment (optional):
    #   THESIS_PYTHON — default python3
    #   THESIS_OUTPUT_DIR — passed as --output-dir
    #
    # Flags after -- are forwarded to export_model_documentation.py, e.g.:
    #   bash scripts/run_export_model_documentation.sh -- --mirror-docs
    
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$ROOT"
    

## `run_export_temp_path_inventory.sh`

    #!/usr/bin/env bash
    # Export recursive path inventory → output/path/
    #
    # From TEMP (repo) root:
    #   bash scripts/run_export_temp_path_inventory.sh
    #
    # Environment (optional):
    #   THESIS_PYTHON — default python3
    #   THESIS_PATH_OUTPUT_DIR — passed as --output-dir
    #
    # Flags after -- are forwarded to export_temp_path_inventory.py, e.g.:
    #   bash scripts/run_export_temp_path_inventory.sh -- --no-default-excludes
    
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$ROOT"
    

## `run_featenc_transformed_embeddings_4gpu_detach.sh`

    #!/usr/bin/env bash
    # FeatureEncoderClassifier on 100-D transformed embeddings: 4× GPU DDP via torchrun + train_queue.
    #
    # Data:   data/transformed/all-data.parquet (columns features_100d + sentiment_value; full row set unless THESIS_MAX_SAMPLES).
    # Models: Code/thesis/config/feature_encoder/{2_labels,3_labels}/FeatEnc_*.json (CNN, LSTM, GRU, RNN, FFNN).
    # Out:    REAL finetune weights (safetensors) under:
    #            $CHECKPOINT_ROOT/{2,3}-labels/all-data/FeatEnc_<ARCH>.safetensors
    #          Example: checkpoints/3-labels/all-data/FeatEnc_CNN.safetensors
    #          Then (unless THESIS_FEATENC_INCLUDE_ML_BC=0) docs/ml Tracks B/C under:
    #            $CHECKPOINT_ROOT/moe/ml_stack/{2,3}-labels/all-data/trackB_*.safetensors and proc_*.joblib
    #          Logs: $LOG_DIR (queue + per-job logs under queue_cnn_rnn_*/).
    # Resume: live bundles under THESIS_RESUME_TEMP (default ./logs/resume) + train_queue state under logs/queue_cnn_rnn_*/queue_state.json
    #
    # Defaults: per-GPU batch 4096, 2 finetune epochs, 4 processes, periodic resume save every 200 steps (override via env).
    #
    # Usage:
    #   cd /path/to/TEMP && bash scripts/run_featenc_transformed_embeddings_4gpu_detach.sh
    # Foreground (no detach):  THESIS_DETACH=none bash scripts/run_featenc_transformed_embeddings_4gpu_detach.sh
    # Resume queue after crash: export THESIS_QUEUE_RESUME_DIR=/path/to/logs/queue_cnn_rnn_<id>  then re-run this script.
    #

## `run_feature_encoder_finetune_track.sh`

    #!/usr/bin/env bash
    # FeatureEncoderClassifier on UMAP 100-D features: CNN, LSTM, GRU, RNN, FFNN (100→100 latent + K-way head).
    # Configs: Code/thesis/config/feature_encoder/{2_labels,3_labels}/FeatEnc_*.json
    # Data: data/transformed/{dataset_stem}.parquet column features_100d (+ sentiment_value).
    #
    # Docs: TEMP/docs/plans/feature_models_all-data_pretrain_e9470305.plan.md
    #       TEMP/docs/thesis_parameter_counts.md (FeatEnc_* rows)
    #       TEMP/docs/ml/TRAINING_PIPELINES.md §3 (same parquet as Track A tabular)
    #
    # Invokes Code/thesis/train/train_queue.py --feature-encoder-only so cnn/B9_CNN_Text and
    # rnn/B7_BiLSTM-style configs are NOT queued (those are separate text-sequence models).
    #
    # Usage (cwd = repo root, usually TEMP):
    #   bash scripts/run_feature_encoder_finetune_track.sh
    #
    # Env:
    #   THESIS_QUEUE_INCLUDE_ALL_DATA=1   — default on; required if only all-data.parquet exists under transformed/
    #   THESIS_FEATENC_SKIP_PRETRAIN=1    — skip five FeaturePretrainAutoencoder runs; finetune encoders from init
    #   THESIS_FEATENC_EPOCHS=2           — --epochs_finetune (default 2; was 8 historically)
    #   THESIS_BATCH_SIZE                 — per-process batch for torchrun (default 24 in train_queue)

## `run_hrm_blackwell_2x96gb_detach.sh`

    #!/usr/bin/env bash
    # HRM MLM pretrain for 2x RTX Pro 6000 Blackwell-class (96GB): DDP, per-GPU batch 64, 2 epochs.
    #
    # Outputs (under TEMP repo root):
    #   checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors
    #   checkpoints/hrm/tokenizer/          (google-bert/bert-base-uncased via HuggingFace save_pretrained)
    #   logs/hrm_blackwell_pretrain_<UTC>.log  (stdout+stderr; default one file per launch)
    #   logs/hrm_blackwell_gpu_usage_<UTC>.log (nvidia-smi samples; pair matches training log)
    #   logs/resume/                        live resume bundles
    #
    # Data: data/processed/all-data.parquet (build: python Code/thesis/data/merge_all_data_parquet.py)
    #
    # Usage:
    #   cd /path/to/TEMP && bash scripts/run_hrm_blackwell_2x96gb_detach.sh
    #
    # tmux:   tmux attach -t hrm_blackwell_pretrain
    # follow: stderr prints exact paths; or: ls -t logs/hrm_blackwell_pretrain_*.log | head -1
    #
    # Tunables (env): THESIS_HRM_BATCH, THESIS_EPOCHS_PRETRAIN, THESIS_NUM_WORKERS,
    #   THESIS_SESSION, THESIS_AMP_BF16 (default on), THESIS_DATALOADER_PERSISTENT (default on)

## `run_hrm_encoder_pretrain_only.sh`

    #!/usr/bin/env bash
    # HRM encoder-only MLM pretrain ONLY (no CNN/RNN/feature queue, no finetune).
    # Writes: {THESIS_CHECKPOINT_ROOT:-checkpoints}/pretrain/{stem}/E_HRM1_4Level.safetensors
    #         + tokenizer at .../tokenizer/ (bert-base-uncased via save_pretrained)
    # Data:    data/processed/all-data.parquet (text column) unless you override stem/source.
    #   If missing, build it: python Code/thesis/data/merge_all_data_parquet.py
    #   See data/processed/README.md
    #
    # Repo root:  bash scripts/run_hrm_encoder_pretrain_only.sh
    #
    # Env (same knobs as the HRM block in run_thesis_pretrain.sh):
    #   THESIS_PYTHON, THESIS_DETACH=none|tmux|screen|nohup, THESIS_SESSION
    #   CUDA_VISIBLE_DEVICES  — e.g. 0 (laptop) or 0,1 (2 GPUs); must match nproc count
    #   THESIS_NPROC_PER_NODE — default: number of commas in CUDA_VISIBLE_DEVICES (min 1)
    #   THESIS_HRM_BATCH — per-GPU batch; unset uses VRAM tiers (2 / 8 / 24 / 64 MB thresholds).
    #     Cloud / 96GB: default tier is already 64; for a fixed batch regardless of tier use:
    #       export THESIS_HRM_BATCH=64
    #     Laptop / 8GB: use export THESIS_HRM_BATCH=1 (or 2) to avoid OOM; do not force 64.
    #   THESIS_NUM_WORKERS, THESIS_GC_EVERY
    #   THESIS_SAVE_EVERY_MINUTES, THESIS_SAVE_EVERY_STEPS, THESIS_MIN_SAVE_INTERVAL_SEC

## `run_hrm_finetune_mlp.sh`

    #!/usr/bin/env bash
    # HRM frozen-encoder supervised finetune with MLP sentiment head (2- or 3-class).
    # Checkpoints: {THESIS_CHECKPOINT_ROOT:-checkpoints/hrm}/fine-tune/{stem}/{K-labels}/E_HRM1_4Level_ft_mlp_{K}label.safetensors
    # Encoder weights: --hrm_encoder_ckpt (default checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors)
    #
    # 4× L40 / L40S production: scripts/run_hrm_finetune_mlp_4xl40.sh [2|3] (CUDA 0–3, nproc=4, default batch 32).
    # Sequential 3-label then 2-label: scripts/run_hrm_finetune_mlp_4xl40_sequential.sh — wrap the whole script in one tmux for detach.
    # 3-label (2 ep) then 2-label (2 ep), separate logs: scripts/run_hrm_finetune_mlp_3then2_two_epochs.sh
    # 2-label pos/neg only from data/processed/all-data.parquet: scripts/run_hrm_finetune_mlp_2label_posneg_all_data.sh
    #
    # Repo root:  bash scripts/run_hrm_finetune_mlp.sh [2|3]
    #   or:        THESIS_HRM_FINETUNE_LABELS=2 bash scripts/run_hrm_finetune_mlp.sh
    #
    # Default: trains on **all** rows in data/processed/{stem}.parquet (no --max_samples unless THESIS_MAX_SAMPLES is set).
    # For stem all-data: parquet has text, sentiment_value, source_stem; sentiment_value on disk is **{0,1,2}**
    # (canonical 3-way). Re-merge normalization: Code/thesis/data/rewrite_all_data_sentiment_three_class.py
    # Default detach: **tmux** (background session). Foreground: THESIS_DETACH=none bash scripts/run_hrm_finetune_mlp.sh 3
    #
    # GPU targets (per-GPU --batch_size; tune with THESIS_HRM_FINETUNE_BATCH or THESIS_HRM_BATCH):
    #   - 2× RTX Pro 6000 Blackwell 96GB: try 32–64 (defaults below use same VRAM tiers as encoder pretrain).

## `run_hrm_finetune_mlp_2label.sh`

    #!/usr/bin/env bash
    # Thin wrapper: 2-label HRM MLP finetune (drops neutral rows). See run_hrm_finetune_mlp.sh.
    exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_hrm_finetune_mlp.sh" 2

## `run_hrm_finetune_mlp_2label_posneg_4xl40s.sh`

    #!/usr/bin/env bash
    # 4× L40 / L40S (~50GB): 2-label HRM MLP finetune on **negative + positive only** (neutral dropped)
    # from data/processed/all-data.parquet. Distributed DDP, per-GPU batch 256 (override THESIS_HRM_FINETUNE_BATCH), 2 epochs, periodic GC.
    #
    #   bash scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
    # Foreground: THESIS_DETACH=none bash scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
    # Fresh run (ignore live resume): THESIS_NO_RESUME=1 bash scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
    #
    # THESIS_DETACH defaults to **none** so 4-GPU / batch / epoch env is not lost when run_hrm_finetune_mlp.sh
    # would otherwise spawn a nested tmux with a bare __inner (no exports). Wrap this script in your own tmux.
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    
    # Do not use ${VAR:-default}: a parent shell often exports CUDA_VISIBLE_DEVICES=0 and would hide all other GPUs.
    export CUDA_VISIBLE_DEVICES="${CUDA_FOUR_GPUS:-0,1,2,3}"
    export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-4}"
    unset THESIS_MAX_SAMPLES 2>/dev/null || true
    unset THESIS_HRM_BATCH 2>/dev/null || true

## `run_hrm_finetune_mlp_2label_posneg_all_data.sh`

    #!/usr/bin/env bash
    # HRM MLP finetune: **2 classes (negative / positive only)** from processed all-data parquet.
    #
    # Data:  $ROOT/data/processed/all-data.parquet  (override with THESIS_DATA_ROOT)
    # Rows:  sentiment_value 0 = negative, 1 = positive, 2 = neutral → **neutral rows are dropped**
    #        (train_single ParquetTextDataset + default 2-class HRM path; do NOT pass --no_hrm_exclude_neutral).
    #
    # Encoder: checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors (override THESIS_HRM_ENCODER_CKPT)
    # Output:  checkpoints/hrm/fine-tune/all-data/2-labels/E_HRM1_4Level_ft_mlp_2label.safetensors (on full success)
    # Live:    logs/resume/E_HRM1_4Level_ft_mlp_2label__all-data__2l/
    # Log:     logs/hrm_finetune_mlp_2label.log
    #
    # 4× L40S (batch 32, 2 ep, GC): scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
    #
    # Usage:
    #   cd /path/to/TEMP && bash scripts/run_hrm_finetune_mlp_2label_posneg_all_data.sh
    # Foreground:  THESIS_DETACH=none bash scripts/run_hrm_finetune_mlp_2label_posneg_all_data.sh
    # GPUs:        CUDA_VISIBLE_DEVICES=0,1,2,3  (must match nproc)
    # Epochs:      THESIS_EPOCHS_FINETUNE=3  (default in run_hrm_finetune_mlp.sh if unset)
    

## `run_hrm_finetune_mlp_3label.sh`

    #!/usr/bin/env bash
    # Thin wrapper: 3-label HRM MLP finetune. See run_hrm_finetune_mlp.sh.
    exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_hrm_finetune_mlp.sh" 3

## `run_hrm_finetune_mlp_3then2_two_epochs.sh`

    #!/usr/bin/env bash
    # 3-label MLP finetune (2 epochs) → clean exit + final .safetensors → 2-label (2 epochs).
    # Logs: logs/hrm_finetune_mlp_3label.log and logs/hrm_finetune_mlp_2label.log
    #
    # Default: THESIS_DETACH=none inside phases so this script blocks (wrap in one outer tmux for SSH-safe runs).
    #   cd /path/to/TEMP && bash scripts/run_hrm_finetune_mlp_3then2_two_epochs.sh
    #   tmux new-session -d -s hrm_mlp_3then2 "cd /path/to/TEMP && bash scripts/run_hrm_finetune_mlp_3then2_two_epochs.sh"
    #
    # Env:
    #   THESIS_EPOCHS_FINETUNE_3LABEL, THESIS_EPOCHS_FINETUNE_2LABEL (default 2 each)
    #   THESIS_PYTHON, CUDA_VISIBLE_DEVICES, THESIS_HRM_FINETUNE_BATCH, THESIS_HRM_FINETUNE_SHARDED, etc. (passed through)
    #   THESIS_NO_RESUME=1 — skip live-resume temp state for each phase (optional)
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$ROOT"
    
    E3="${THESIS_EPOCHS_FINETUNE_3LABEL:-2}"
    E2="${THESIS_EPOCHS_FINETUNE_2LABEL:-2}"

## `run_hrm_finetune_mlp_4xl40.sh`

    #!/usr/bin/env bash
    # 4× NVIDIA L40 / L40S (~48–50GB): pin GPUs 0–3, nproc=4, default per-GPU batch 32 for MLP finetune.
    # Drops any THESIS_MAX_SAMPLES so the full all-data parquet is used.
    #
    # Dataset (default stem all-data): data/processed/all-data.parquet — columns text, sentiment_value,
    # source_stem; sentiment_value is canonical 3-way only {0,1,2} on disk (rewritten). Override stem:
    #   THESIS_HRM_PRETRAIN_STEM=my-stem bash scripts/run_hrm_finetune_mlp_4xl40.sh 3
    # If you re-merge all-data from heterogeneous sources, re-run:
    #   python Code/thesis/data/rewrite_all_data_sentiment_three_class.py
    #
    # Usage: bash scripts/run_hrm_finetune_mlp_4xl40.sh [2|3]
    # On OOM, lower batch: THESIS_HRM_FINETUNE_BATCH=24 bash scripts/run_hrm_finetune_mlp_4xl40.sh 3
    # Sequential 3-label then 2-label: see run_hrm_finetune_mlp_4xl40_sequential.sh
    # If a parent shell exported a low smoke-test THESIS_HRM_FINETUNE_BATCH, use:
    #   env -u THESIS_HRM_FINETUNE_BATCH bash scripts/run_hrm_finetune_mlp_4xl40.sh 3
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    

## `run_hrm_finetune_mlp_4xl40_sequential.sh`

    #!/usr/bin/env bash
    # Production: 3-label MLP finetune first, then 2-label, sequentially on the same 4×L40 box.
    # Both phases use the same frozen encoder; outputs go to fine-tune/all-data/{3,2}-labels/ separately.
    # Epochs per phase: THESIS_EPOCHS_FINETUNE (default 3; set to 2 for shorter runs).
    #
    # Data: default THESIS_HRM_PRETRAIN_STEM=all-data → data/processed/all-data.parquet with
    # sentiment_value ∈ {0,1,2} only (see run_hrm_finetune_mlp_4xl40.sh header). 2-label phase drops
    # neutral (2) rows per train_single / ParquetTextDataset.
    #
    # Detach both phases together with one outer tmux (avoid nested tmux per run):
    #   tmux new-session -d -s hrm_mlp_finetune_seq \
    #     "cd /path/to/TEMP && export THESIS_PYTHON=... THESIS_EPOCHS_FINETUNE=3 && bash scripts/run_hrm_finetune_mlp_4xl40_sequential.sh"
    #   tmux attach -t hrm_mlp_finetune_seq
    # Tip: if your environment exports a stale THESIS_HRM_FINETUNE_BATCH, prefix with env -u THESIS_HRM_FINETUNE_BATCH
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    export THESIS_EPOCHS_FINETUNE="${THESIS_EPOCHS_FINETUNE:-3}"
    export THESIS_HRM_PRETRAIN_STEM="${THESIS_HRM_PRETRAIN_STEM:-all-data}"

## `run_hrm_finetune_mlp_4xl40_sequential_with_dotenv.sh`

    #!/usr/bin/env bash
    # Sequential 4×L40 finetune with HF_TOKEN / HUGGINGFACE_HUB_TOKEN from repo-root .env (if present).
    # Does not print the token.
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    ENV_FILE="$ROOT/.env"
    if [[ -f "$ENV_FILE" ]]; then
      PY="${ROOT}/.venv/bin/python"
      [[ -x "$PY" ]] || PY=python3
      HF_TOKEN="$("$PY" -c 'import re, sys
    from pathlib import Path
    for line in Path(sys.argv[1]).read_text().splitlines():
        m = re.match("^\\s*HF_TOKEN\\s*=\\s*(.+)$", line)
        if m:
            print(m.group(1).strip())
            break
    ' "$ENV_FILE")"
      if [[ -n "${HF_TOKEN:-}" ]]; then

## `run_ml_bc_before_moe.sh`

    #!/usr/bin/env bash
    # Run docs/ml Track B or C (train_ml_processed_embed_meta.py) before MoE gate training.
    # Writes under checkpoints/moe/ml_stack/{2,3}-labels/{stem}/ per TRAINING_PIPELINES.md §4–5.
    #
    # Usage:
    #   TRACK=b N_LABELS=2 STEM=all-data bash scripts/run_ml_bc_before_moe.sh
    #   TRACK=c N_LABELS=3 STEM=all-data DATA_ROOT=/path/to/data bash scripts/run_ml_bc_before_moe.sh
    #
    # Env: ROOT (default: TEMP repo), TRACK (b|c), N_LABELS (2|3), STEM, DATA_ROOT, CHECKPOINT_ROOT, EXTRA_ARGS
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$ROOT"
    export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
    
    TRACK="${TRACK:-b}"
    N_LABELS="${N_LABELS:-2}"
    STEM="${STEM:-all-data}"
    QUEUE="$ROOT/Code/thesis/config/ml_queue/track_${TRACK}_${N_LABELS}_labels.json"
    if [[ ! -f "$QUEUE" ]]; then
      echo "Missing queue config: $QUEUE" >&2

## `run_ml_tabular_track_a.sh`

    #!/usr/bin/env bash
    # docs/ml Track A: train E_ML1–E_ML4 (LR, LinearSVC, DecisionTree, RandomForest) on
    # data/transformed/{dataset_stem}.parquet — default dataset_stem=all-data.
    # DT/RF checkpoints are .joblib (full state); LR/SVC remain .safetensors.
    #
    # Usage (from TEMP = repo root):
    #   bash scripts/run_ml_tabular_track_a.sh
    #   THESIS_DATA_ROOT=... THESIS_PYTHON=... bash scripts/run_ml_tabular_track_a.sh
    #
    # Logs: logs/ml_tabular_track_a.log (override THESIS_ML_TABULAR_LOG) plus per-job
    # logs/ml_tabular_{dataset_stem}_{2|3}label_{config_stem}.log under logs/.
    # Optional: THESIS_MAX_SAMPLES — passed as --max_samples to train_single (smoke / subsample).
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$ROOT"
    PY="${THESIS_PYTHON:-python3}"
    [[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
    

## `run_mlp_geLU_head_all_configs_3gpu.sh`

    #!/usr/bin/env bash
    # Run every thesis config under config/mlp_gelu_head_ddp/{2_labels,3_labels}/ (FFN, CNN, LSTM, GRU, RNN).
    # Uses physical GPUs 1,2,3 only (see CUDA_VISIBLE_DEVICES).
    set -euo pipefail
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
    PORT="${MASTER_PORT_BASE:-29550}"
    CFG_ROOT="${ROOT}/Code/thesis/config/mlp_gelu_head_ddp"
    AGG="${ROOT}/logs/mlp_geLU_head_ddp_ALL_CONFIGS_$(date -u +%Y%m%d_%H%M%S).log"
    echo "Aggregate log: $AGG"
    {
      echo "======== ALL mlp_gelu_head_ddp configs ========"
      for LABEL_DIR in "${CFG_ROOT}/2_labels" "${CFG_ROOT}/3_labels"; do
        [[ -d "$LABEL_DIR" ]] || continue
        for cfg in "${LABEL_DIR}"/*.json; do
          [[ -f "$cfg" ]] || continue
          echo ""
          echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
          echo "$(date -u -Iseconds) START $cfg port=$PORT"

## `run_mlp_geLU_head_finetune_3gpu.sh`

    #!/usr/bin/env bash
    # Fine-tune GeLU head on GPUs 1, 2, 3 (physical indices after CUDA_VISIBLE_DEVICES remap to 0,1,2).
    set -euo pipefail
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
    N_CLASSES="${1:?usage: $0 <2|3> [extra torchrun/train args...]}"
    shift
    PORT="${MASTER_PORT:-29531}"
    exec torchrun --nproc_per_node=3 --master_port="${PORT}" \
      "${ROOT}/Code/thesis/train/train_frozen_pretrain_mlp_head_ddp.py" \
      --data_parquet "${ROOT}/data/transformed/all-data.parquet" \
      --checkpoint_root "${ROOT}/checkpoints" \
      --log_dir "${ROOT}/logs" \
      --pretrain_arch ffnn \
      --n_classes "${N_CLASSES}" \
      --epochs 2 \
      --batch_size 512 \
      --num_workers 4 \
      "$@"

## `run_moe_expert_matrix.sh`

    #!/usr/bin/env bash
    # Run train_moe for every expert manifest under Code/thesis/config/moe/ (excluding example_experts.json).
    # Intended: 2-label and 3-label all-data configs; separate log per manifest under TEMP/logs.
    #
    # Defaults (full merged corpus):
    #   DATA_ROOT=$ROOT/data
    #   DATASET_STEM=all-data
    #   No max_samples cap (omit --max_samples)
    #
    # Env:
    #   MOE_MATRIX_PROFILE — all_data (default: dense feature-encoder experts only, 2+3 label) or
    #     with_distilbert (DistilBERT + FFN like old all_data) or full (every experts_*.json except example/with_distilbert)
    #   MOE_MANIFEST_LIST — space-separated basenames or paths (optional); default = all experts_*.json
    #   MOE_SKIP_MANIFESTS — regex or space list to skip (optional)
    #   VERIFY_EXPERT_CHECKPOINTS — 1 validate ckpts (default 1)
    #   NPROC — torchrun ranks (default 1)
    #   EPOCHS BATCH_SIZE SAVE_EVERY_STEPS AUTO_BATCH_VRAM_TARGET RESUME EXTRA_ARGS
    #   FLUSH_LOGS — if 1, rm TEMP/logs/moe_matrix_* before run
    #   SYNC_LABELS — default 1 (--sync-labels)
    #

## `run_moe_full_all_gpus.sh`

    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$ROOT"
    export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
    
    mkdir -p "$ROOT/logs"
    TS="$(date +%Y%m%d_%H%M%S)"
    GPU_LOG="$ROOT/logs/nvidia_smi_moe_full_${TS}.log"
    GPU_INTERVAL="${GPU_INTERVAL:-5}"
    
    cleanup() {
      if [[ -n "${GPU_MON_PID:-}" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
        kill "$GPU_MON_PID" 2>/dev/null || true
      fi
    }
    trap cleanup EXIT
    
    if command -v nvidia-smi >/dev/null 2>&1 && [[ "${GPU_MONITOR:-1}" == "1" ]]; then
      {

## `run_moe_qlora_smoke.sh`

    #!/usr/bin/env bash
    # MoE gate smoke: optional torchrun DDP, tiny LoRA preset, logs under ROOT/logs only.
    #
    # Dual L40S (example):
    #   export CUDA_VISIBLE_DEVICES=0,1
    #   export MASTER_ADDR=127.0.0.1
    #   export MASTER_PORT=29505
    #   NPROC=2 bash scripts/run_moe_qlora_smoke.sh
    #
    # Single GPU / CPU:
    #   NPROC=1 bash scripts/run_moe_qlora_smoke.sh
    #
    # Env:
    #   EXPERTS   — path to experts JSON (default: Code/thesis/config/moe/experts_smoke_2label.json)
    #   NPROC     — torchrun --nproc_per_node (default 1; use 2 on dual L40S)
    #   SYNTHETIC — if 1 and DUMMY parquets missing, use synthetic data
    #   EXTRA_ARGS — extra CLI passed to train_moe.py
    #   VERIFY_EXPERT_CHECKPOINTS — set to 0 to skip JSON checkpoint file checks (random-init experts)
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

## `run_moe_training_resilient.sh`

    #!/usr/bin/env bash
    # Full MoE run: flush TEMP/logs, background GPU sampling, periodic checkpoints, optional VRAM-target batch size.
    #
    #   FLUSH_LOGS=1              — remove all files in ROOT/logs (default 1)
    #   GPU_MONITOR=1             — append nvidia-smi CSV every GPU_INTERVAL sec (default 1)
    #   GPU_INTERVAL=5
    #   EPOCHS=1
    #   SAVE_EVERY_STEPS=100
    #   AUTO_BATCH_VRAM_TARGET=0.8 — omit to use fixed BATCH_SIZE
    #   RESUME=1                  — pass --resume to train_moe
    #   NPROC=2
    #   SYNTHETIC=1 / VERIFY_EXPERT_CHECKPOINTS=0
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$ROOT"
    export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
    
    MOE_REPO_ROOT="$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/scripts/moe_ddp_launch.sh"

## `run_per_source_stem_metrics.sh`

    #!/usr/bin/env bash
    # Per-source_stem split + checkpoint evaluation → output/metrics/
    #
    # From TEMP (repo) root:
    #   bash scripts/run_per_source_stem_metrics.sh
    #
    # Environment (optional):
    #   THESIS_PYTHON   — default python3
    #   THESIS_DATA_ROOT, THESIS_CHECKPOINT_ROOT — passed as --data-root / --checkpoint-root
    #
    # Full split on large all-data can take a long time and needs RAM. For smoke tests:
    #   THESIS_SPLIT_MAX_ROWS=5000 THESIS_MAX_SAMPLES=32 bash scripts/run_per_source_stem_metrics.sh
    #
    # After a run, flat tables are written (and refreshed from all metrics.json):
    #   output/metrics/2label_metrics_table.csv
    #   output/metrics/3label_metrics_table.csv
    # Rebuild only those CSVs from existing JSON:
    #   PYTHONPATH=. python3 Code/thesis/tools/eval_per_source_stem_metrics.py --export-metrics-csv-only
    #
    # Flags after -- are forwarded to eval_per_source_stem_metrics.py (e.g. --only-stems a,b)

## `run_thesis_finetune_text.sh`

    #!/usr/bin/env bash
    # Frozen-backbone finetune for transformers + HRM (processed parquets except all-data).
    # HRM uses Code/thesis/config/hrm/E_HRM1_4Level.json with --n_classes 2 then 3 per stem.
    # Repo root: bash scripts/run_thesis_finetune_text.sh
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$ROOT"
    PY="${THESIS_PYTHON:-python3}"
    
    export THESIS_RESUME_TEMP="${THESIS_RESUME_TEMP:-$ROOT/logs/resume}"
    mkdir -p "$ROOT/logs" "$THESIS_RESUME_TEMP"
    
    NPROC="${THESIS_NPROC_PER_NODE:-2}"
    TXT_BS="${THESIS_TEXT_FINETUNE_BATCH:-4}"
    
    mapfile -t CONFIGS < <(find "$ROOT/Code/thesis/config/transformers" "$ROOT/Code/thesis/config/hrm" -name '*.json' 2>/dev/null | sort -u)
    
    run_inner() {

## `run_thesis_pretrain.sh`

    #!/usr/bin/env bash
    # HRM MLM on data/processed/all-data.parquet (DDP), then CNN/RNN/ML queue.
    # Repo root:  bash scripts/run_thesis_pretrain.sh
    #
    # Env (high level):
    #   THESIS_PYTHON, THESIS_DETACH=none|tmux|screen|nohup, THESIS_SESSION
    #   CUDA_VISIBLE_DEVICES — e.g. 0,1 for 2 GPUs or 0,1,2,3 for 4 (must match nproc)
    #   THESIS_NPROC_PER_NODE — defaults to comma-count of CUDA_VISIBLE_DEVICES (after default 0,1)
    #   THESIS_HRM_BATCH, THESIS_FEAT_BATCH — override per-GPU batch sizes
    #   HRM E-HRM1 uses seq_len 512 (long-context attention): if you OOM, set THESIS_HRM_BATCH lower than the VRAM-tier defaults below.
    #   THESIS_NUM_WORKERS (default 8), THESIS_GC_EVERY (default 0)
    #   THESIS_SAVE_EVERY_MINUTES (default 5), THESIS_SAVE_EVERY_STEPS (default 1000)
    #   THESIS_MIN_SAVE_INTERVAL_SEC (default 45)
    #   (Removed) THESIS_HRM_PRETRAIN_3LABEL — redundant for encoder-only MLM; finetune 2/3-way uses --n_classes separately.
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$ROOT"
    PY="${THESIS_PYTHON:-python3}"

## `run_transformer_mlp_finetune.sh`

    #!/usr/bin/env bash
    # Frozen Hugging Face transformer backbone + trainable MLP head (768→1024→GELU→K).
    # Checkpoints: {THESIS_CHECKPOINT_ROOT:-checkpoints}/{K}-labels/{stem}/{config_stem}.safetensors
    #
    # Usage (from repo root = directory containing Code/):
    #   bash scripts/run_transformer_mlp_finetune.sh [2|3] [config_stem]
    #   THESIS_TRANSFORMER_CFG_STEM=B4_E_DL3_BERT_mlp768_1024 bash scripts/run_transformer_mlp_finetune.sh 2
    #   THESIS_TRANSFORMER_RUN_ALL=1 THESIS_DETACH=none bash scripts/run_transformer_mlp_finetune.sh 3
    #
    # 4× GPU: scripts/run_transformer_mlp_finetune_4xl40s.sh [2|3] [config_stem]
    # All 8 (B3–B6 × 3 then 2), foreground: scripts/run_transformer_mlp_finetune_4xl40s_sequential_all8.sh
    #
    # Env: THESIS_PYTHON, THESIS_DETACH=tmux|none|screen|nohup (default tmux), THESIS_SESSION
    #   CUDA_VISIBLE_DEVICES, THESIS_NPROC_PER_NODE
    #   THESIS_TRANSFORMER_FINETUNE_BATCH (default 128 per GPU here; 4xl40s wrapper may override), THESIS_EPOCHS_FINETUNE (default 2)
    #   THESIS_LR (default 1e-3, head-only)
    #   THESIS_DATA_ROOT, THESIS_TRANSFORMER_DATASET_STEM (default all-data)
    #   THESIS_CHECKPOINT_ROOT, THESIS_RESUME_TEMP, THESIS_NO_RESUME
    #   THESIS_SAVE_EVERY_*, THESIS_NUM_WORKERS, THESIS_GC_EVERY
    #   THESIS_TRANSFORMER_FINETUNE_LOG, THESIS_MAX_SAMPLES

## `run_transformer_mlp_finetune_2label.sh`

    #!/usr/bin/env bash
    # Wrapper: 2-class transformer MLP head finetune.
    exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_transformer_mlp_finetune.sh" 2 "$@"

## `run_transformer_mlp_finetune_3label.sh`

    #!/usr/bin/env bash
    # Wrapper: 3-class transformer MLP head finetune.
    exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_transformer_mlp_finetune.sh" 3 "$@"

## `run_transformer_mlp_finetune_4xl40s.sh`

    #!/usr/bin/env bash
    # 4× GPU defaults for L40/L40S: per-GPU batch 128, 2 epochs (override THESIS_TRANSFORMER_FINETUNE_BATCH).
    # Chain all eight finetunes: run_transformer_mlp_finetune_4xl40s_sequential_all8.sh
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-4}"
    export THESIS_TRANSFORMER_FINETUNE_BATCH="${THESIS_TRANSFORMER_FINETUNE_BATCH:-128}"
    export THESIS_EPOCHS_FINETUNE="${THESIS_EPOCHS_FINETUNE:-2}"
    export THESIS_TORCH_MASTER_PORT="${THESIS_TORCH_MASTER_PORT:-29601}"
    exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_transformer_mlp_finetune.sh" "$@"

## `run_transformer_mlp_finetune_4xl40s_sequential_all8.sh`

    #!/usr/bin/env bash
    # 4× L40/L40S: run all eight transformer MLP finetunes in one foreground pipeline —
    # B3–B6 configs × 3-label, then the same four × 2-label.
    #
    # Same resource defaults as run_transformer_mlp_finetune_4xl40s.sh:
    #   CUDA 0–3, nproc=4, per-GPU batch 128, 2 finetune epochs (override via env).
    # Uses THESIS_TRANSFORMER_RUN_ALL=1 twice (labels 3 then 2); THESIS_DETACH=none so
    # this script blocks (wrap in tmux/screen at the caller for SSH-safe long runs).
    #
    # Resume: inherits periodic saves from run_transformer_mlp_finetune.sh unless THESIS_NO_RESUME=1.
    #
    # Usage (repo root = TEMP, directory containing Code/):
    #   bash scripts/run_transformer_mlp_finetune_4xl40s_sequential_all8.sh
    # After HRM 2-label finetune (GPU idle): scripts/queue_transformer_mlp_all8_after_hrm_2label.sh
    #
    # Env: same as scripts/run_transformer_mlp_finetune.sh (HF_TOKEN, THESIS_PYTHON,
    #   THESIS_DATA_ROOT, THESIS_CHECKPOINT_ROOT, THESIS_SAVE_EVERY_*, etc.)
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

## `run_transformer_mlp_finetune_smoke.sh`

    #!/usr/bin/env bash
    # Quick sanity run: few rows, 1 epoch, 1 GPU, small batch. For local/CI checks only.
    #
    # Defaults (override via env):
    #   THESIS_MAX_SAMPLES=128
    #   THESIS_EPOCHS_FINETUNE=1
    #   THESIS_TRANSFORMER_FINETUNE_BATCH=8
    #   CUDA_VISIBLE_DEVICES=0, THESIS_NPROC_PER_NODE=1
    #   THESIS_TORCH_MASTER_PORT=29521  (avoid clashing with a full DDP job on 29500)
    #   THESIS_DETACH=none
    #
    # Usage (from TEMP/):
    #   bash scripts/run_transformer_mlp_finetune_smoke.sh 3
    #   bash scripts/run_transformer_mlp_finetune_smoke.sh 2 B4_E_DL3_BERT_mlp768_1024
    
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [[ -z "${THESIS_PYTHON:-}" ]]; then
      if [[ -x "$ROOT/.venv/bin/python" ]]; then

## `smoke_hrm_ddp_sigterm_resume.sh`

    #!/usr/bin/env bash
    # Cloud / dual-GPU: HRM encoder-only MLM — interrupt mid-epoch and resume (DDP).
    # Repo root. Requires: 2 GPUs, same env as run_hrm_encoder_pretrain_only.sh, HF cache.
    #
    # This script does NOT auto-kill (too environment-specific). Copy-paste pattern:
    #
    # 1) Terminal A — start training (tmux recommended):
    #    export CUDA_VISIBLE_DEVICES=0,1
    #    export THESIS_EPOCHS_PRETRAIN=1
    #    export THESIS_SAVE_EVERY_STEPS=200
    #    export THESIS_DETACH=tmux
    #    export THESIS_SESSION=hrm_ddp_resume_test
    #    bash scripts/run_hrm_encoder_pretrain_only.sh
    #
    # 2) After you see several hundred steps in logs/hrm_encoder_pretrain_only.log, send SIGTERM to the
    #    train_single parent (or close tmux pane). Verify logs/resume/.../meta.json exists.
    #
    # 3) Relaunch the SAME command (same THESIS_RESUME_TEMP, data_root, checkpoint_root, stem).
    #    Training should skip completed batches and continue; final artifact:
    #    checkpoints/pretrain/<stem>/E_HRM1_4Level.safetensors

## `stop_all_temp_training.sh`

    #!/usr/bin/env bash
    # Stop all thesis training processes rooted at this TEMP checkout (torchrun + train_single + train_queue).
    # Invoked as a file so interactive tools don't match pkill -f on their own command line.
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    PY="$ROOT/.venv/bin/python"
    for sig in TERM TERM KILL; do
      pkill -$sig -f "${PY}.*train_queue\.py" 2>/dev/null || true
      pkill -$sig -f "${PY}.*torch\.distributed\.run.*train_single" 2>/dev/null || true
      pkill -$sig -f "${PY} -u .*train_single\.py" 2>/dev/null || true
      sleep 1
    done
    echo "[stop-temp-train] done (best-effort)."

## `tmux_moe_qlora_smoke.sh`

    #!/usr/bin/env bash
    # Detached tmux session running MoE QLoRA/smoke training (see run_moe_qlora_smoke.sh).
    # Usage: bash scripts/tmux_moe_qlora_smoke.sh
    # Attach: tmux attach -t moe_smoke
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    SESSION="${MOE_TMUX_SESSION:-moe_smoke}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION" >&2
      exit 1
    fi
    tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash ./scripts/run_moe_qlora_smoke.sh; echo; echo 'Done. Press Enter to close.'; read _"
    echo "Started tmux session: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
    echo "Logs:   $ROOT/logs/moe_smoke_*.log"
