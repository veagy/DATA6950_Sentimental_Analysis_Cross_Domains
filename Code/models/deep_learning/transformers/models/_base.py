"""
Base class and shared helpers for end-to-end transformer pipeline models.

All concrete models (DecoderLM, EncoderLM, TranslationalLM) inherit from
BaseTransformerModel, which provides:
  - _resolve_pipeline_module  — priority-based module resolution
  - _parse_model_config       — normalises the top-level model_config dict
  - _build_transformer_layers — creates an nn.ModuleList of stacked transformer blocks
  - generate                  — autoregressive token generation loop
  - encode_text               — tokenise → embed → positional-encode → forward pass
"""

import warnings
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union

from .....models.utils import DLModule
from .....config.deep_learning import instantiate_model


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_pipeline_module(
    explicit: Optional[nn.Module],
    cfg: Optional[Dict],
) -> Optional[nn.Module]:
    """
    Resolve a single pipeline stage to a concrete ``nn.Module``.

    Priority
    --------
    1. ``explicit`` — returned as-is when not None.
    2. ``cfg`` with a ``"model_name"`` key — instantiated via
       ``instantiate_model(model_name, overrides)``.
    3. None — caller decides the fallback (``nn.Identity`` or raise).
    """
    if explicit is not None:
        return explicit
    if cfg is not None:
        model_name = cfg.get("model_name")
        if model_name:
            overrides = {k: v for k, v in cfg.items() if k != "model_name"}
            return instantiate_model(model_name, overrides)
    return None


def _parse_model_config(
    model_config: Optional[Union[Dict, Any]],
) -> Dict[str, Any]:
    """
    Normalise the top-level ``model_config`` into a canonical dict.

    Accepts a dict or a config dataclass (from configs.py). Config objects
    are converted via config_to_dict() and merged into the canonical form.

    Canonical keys
    --------------
    ``"tokenizer"``          – tokeniser stage config dict
    ``"embeddings"``         – embedding stage config dict
    ``"positional_encoder"`` – positional-encoding stage config dict
    ``"transformer"``        – shared per-layer transformer block module_config
                               (forwarded verbatim to each ``GeneralTransformer``)
    ``"transformers"``       – optional per-layer override list; broadcast when
                               shorter than n_layers
    ``"logits_head"``        – logits-head stage config dict   (DecoderLM / TranslationalLM)
    ``"token_selector"``     – token-selection stage config dict (DecoderLM / TranslationalLM)
    ``"pooling"``            – pooling stage config dict        (EncoderLM)
    ``"cross_attention"``    – cross-attn stage config dict     (TranslationalLM)
    ``"encoder"``            – sub-dict holding encoder-side overrides (TranslationalLM)
    ``"decoder"``            – sub-dict holding decoder-side overrides (TranslationalLM)

    Any unknown keys are passed through unchanged so callers can extend the dict.
    """
    if model_config is not None and not isinstance(model_config, dict):
        try:
            from .configs import config_to_dict
            model_config = config_to_dict(model_config)
        except ImportError:
            model_config = {}
    _CANONICAL = {
        # language pipeline stages
        "tokenizer", "embeddings", "positional_encoder",
        "transformer", "transformers",
        "logits_head", "token_selector",
        "pooling",
        "cross_attention",
        "encoder", "decoder",
        # vision
        "patch_embed", "cls_token", "image_encoder", "head", "stem",
        "distill_token", "mask_token", "pixel_head", "decoder_embed",
        "patch_norm", "window_size",
        # audio
        "audio_encoder", "audio_tokenizer", "ctc_head", "acoustic_head",
        # multimodal
        "text_encoder", "vision_projector", "image_grounded_text_encoder",
        "perceiver_resampler", "gated_cross_attn",
        # rag
        "retriever", "reader", "query_encoder", "context_encoder",
        # generative / diffusion
        "diffusion_head", "time_embed", "class_embed", "adaLN_modulation",
        "image_stream", "text_stream",
        # reasoning
        "reward_head", "value_head", "base_model",
        # efficient
        "routing", "experts", "ssm_module",
        # graph
        "node_embed", "edge_embed", "graph_bias", "centrality_embed",
        # scientific
        "contact_head", "property_head", "sequence_embed",
        # temporal
        "patch_projection", "forecast_head", "feature_embed",
        # video
        "temporal_embed", "spatial_layers", "temporal_layers",
    }
    result: Dict[str, Any] = {k: None for k in _CANONICAL}
    if model_config:
        for k, v in model_config.items():
            result[k] = v
    return result


def _broadcast_layer_configs(
    shared_cfg: Optional[Dict],
    per_layer_cfgs: Optional[List[Dict]],
    n: int,
) -> List[Optional[Dict]]:
    """
    Return a list of ``n`` per-layer transformer module_config dicts.

    Rules
    -----
    - If ``per_layer_cfgs`` has exactly ``n`` entries, use as-is.
    - If ``per_layer_cfgs`` is shorter than ``n``, repeat the last entry.
    - If ``per_layer_cfgs`` is None, replicate ``shared_cfg`` n times.
    """
    if per_layer_cfgs is not None and len(per_layer_cfgs) > 0:
        if len(per_layer_cfgs) >= n:
            return list(per_layer_cfgs[:n])
        # broadcast: pad with last entry
        return list(per_layer_cfgs) + [per_layer_cfgs[-1]] * (n - len(per_layer_cfgs))
    return [shared_cfg] * n


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseTransformerModel(DLModule):
    """
    Abstract base for ``DecoderLM``, ``EncoderLM``, and ``TranslationalLM``.

    Subclasses are responsible for building their own ``nn.ModuleList`` of
    transformer layers and registering all sub-modules via normal ``nn.Module``
    attribute assignment so PyTorch's ``state_dict`` / ``parameters`` work
    correctly.

    Shared utilities provided here:

    generate(input_ids, max_new_tokens, eos_token_id, pad_token_id)
        Autoregressive greedy/sampling loop (delegated to ``forward``).

    encode_text(text)
        Tokenise → embed → positional-encode pipeline helper.  Returns the
        hidden-state tensor before any transformer layers.

    Parameters
    ----------
    device : str
    dtype  : torch.dtype
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._device = device
        self._dtype = dtype

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _run_tokenizer(
        self,
        x: Union[str, List[str], torch.Tensor],
        tokenizer: Optional[nn.Module],
    ) -> torch.Tensor:
        """
        If ``x`` is already a tensor, return it directly (skip tokenisation).
        Otherwise call ``tokenizer(x)`` and coerce the result to a tensor.
        """
        if isinstance(x, torch.Tensor):
            return x
        if tokenizer is None or isinstance(tokenizer, nn.Identity):
            raise TypeError(
                "Input is a string but no tokenizer is configured. "
                "Pass a pre-tokenised tensor or set tokenizer_module / model_config['tokenizer']."
            )
        out = tokenizer(x)
        if isinstance(out, dict):
            # Many tokenizers return {"input_ids": ..., "attention_mask": ...}
            return out.get("input_ids", next(iter(out.values())))
        if isinstance(out, (list, tuple)):
            return torch.tensor(out, device=self._device)
        return out

    def encode_text(
        self,
        x: Union[str, List[str], torch.Tensor],
        *,
        tokenizer: Optional[nn.Module] = None,
        embeddings: Optional[nn.Module] = None,
        positional_encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Tokenise → embed → positional-encode and return the hidden tensor.

        Falls back to identity if any stage is None / nn.Identity.
        """
        ids = self._run_tokenizer(x, tokenizer)
        hidden = embeddings(ids) if embeddings is not None else ids
        hidden = positional_encoder(hidden) if positional_encoder is not None else hidden
        return hidden

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation loop.

        Calls ``self.forward(input_ids)`` at each step, takes the argmax of
        the last-token logits, appends it, and stops when ``eos_token_id``
        is produced or ``max_new_tokens`` steps are reached.

        Subclasses that need beam search or sampling should override this.
        """
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.forward(generated, **kwargs)
            # out shape: (batch, seq_len, vocab_size) or (batch, vocab_size)
            if out.dim() == 3:
                logits = out[:, -1, :]  # last token
            else:
                logits = out
            next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

    def forward(self, x: Any, **kwargs) -> Any:  # type: ignore[override]
        raise NotImplementedError("Subclasses must implement forward().")
