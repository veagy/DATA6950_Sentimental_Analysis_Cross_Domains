#!/usr/bin/env python3
"""
Download Hugging Face base models + tokenizers into ``checkpoints/transformer/<name>``.

Run from repository root (requires ``transformers``, ``torch``)::

    python Code/thesis/tools/download_transformers.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

_REPO = Path(__file__).resolve().parents[3]

from transformers import AutoModel, AutoTokenizer  # noqa: E402


MODELS: list[tuple[str, str]] = [
    ("distilbert-base-uncased", "distilbert-base-uncased"),
    ("bert-base-uncased", "bert-base-uncased"),
    ("roberta-base", "roberta-base"),
    ("facebook/bart-base", "bart-base"),
]


def main() -> None:
    target_dir = _REPO / "checkpoints" / "transformer"
    target_dir.mkdir(parents=True, exist_ok=True)

    for hf_name, local_name in MODELS:
        print(f"Downloading {hf_name}...")
        model_dir = target_dir / local_name
        model_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        tokenizer.save_pretrained(model_dir)
        print(f"  Saved tokenizer to {model_dir}")

        model = AutoModel.from_pretrained(hf_name)
        model.save_pretrained(model_dir)
        print(f"  Saved model to {model_dir}")

    print("All models downloaded successfully!")


if __name__ == "__main__":
    main()
