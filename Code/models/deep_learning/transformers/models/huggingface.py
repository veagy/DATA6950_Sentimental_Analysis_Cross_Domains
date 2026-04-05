"""Minimal Hugging Face wrapper for ``LLMModule`` (thesis transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


class HuggingFaceTransformer:
    """Thin ``AutoModel`` + ``AutoTokenizer`` holder with ``save_pretrained``."""

    def __init__(
        self,
        hf_model: Any,
        hf_tokenizer: Any,
        model_type: str = "transformers",
    ):
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.model_type = model_type

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        model_type: str = "auto",
        tokenizer_id: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> "HuggingFaceTransformer":
        from transformers import AutoModel, AutoTokenizer

        tok_src = tokenizer_id or model_id
        tokenizer = AutoTokenizer.from_pretrained(tok_src, **kwargs)
        model = AutoModel.from_pretrained(model_id, **kwargs)
        model.to(device=device, dtype=dtype if device != "cpu" else torch.float32)
        internal = "transformers" if model_type != "sentence_transformer" else "sentence_transformers"
        return cls(model, tokenizer, model_type=internal)

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.hf_model.save_pretrained(directory)
        self.hf_tokenizer.save_pretrained(directory)
