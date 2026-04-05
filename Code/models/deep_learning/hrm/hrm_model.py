import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from ....models.utils import DLModule
from ..transformers.models import get_models


def _find_default_local_hrm_tokenizer_dir() -> Optional[Path]:
    """First ancestor of this file that contains ``checkpoints/hrm/tokenizer`` with files."""
    start = Path(__file__).resolve().parent
    for anc in (start, *start.parents):
        cand = anc / "checkpoints" / "hrm" / "tokenizer"
        try:
            if cand.is_dir() and any(cand.iterdir()):
                return cand
        except OSError:
            continue
    return None


def _resolve_hrm_tokenizer_source(pretrained_id: Optional[str]) -> tuple[Optional[str], bool]:
    """Return ``(path_or_hub_id, local_files_only)`` for ``AutoTokenizer.from_pretrained``."""
    if not pretrained_id:
        return None, False
    p = Path(pretrained_id)
    if p.is_dir():
        return str(p.resolve()), True
    env = (os.environ.get("THESIS_HRM_TOKENIZER_DIR") or "").strip()
    if env:
        ep = Path(env)
        if ep.is_dir():
            return str(ep.resolve()), True
    loc = _find_default_local_hrm_tokenizer_dir()
    if loc is not None:
        return str(loc), True
    return pretrained_id, False

@dataclass
class HRMConfig:
    """Configuration for the Hierarchical Reasoning Model (HRM)."""
    batch_size: int = 1
    seq_len: int = 128
    vocab_size: int = 30522  # BERT wordpiece; overridden when tokenizer loads
    hidden_size: int = 768
    output_embed_dim: int = 100  # sentence embedding dim after pool (encoder-only path)
    H_cycles: int = 2
    L_cycles: int = 3
    halt_max_steps: int = 5
    h_level_model: str = "DecoderLM"
    l_level_model: str = "DecoderLM"
    tokenizer_name: Optional[str] = None
    model_kwargs: dict = field(default_factory=dict)
    
@dataclass
class HRMInnerCarry:
    """State storage for the inner recurrent computation loop."""
    z_H: torch.Tensor
    z_L: torch.Tensor

def _inner_encoder_lm_kwargs(
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    seq_len: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Build EncoderLM kwargs so transformer blocks are real (not Identity).

    Uses ``GeneralAttentionBlock`` (registered in ``model_config_registry.json``);
    ``MultiHeadAttention`` is not a registry key in this codebase.

    The FFN is a single ``StandardDense`` with ``out_features == hidden_size`` so
    the residual in ``GeneralTransformer`` stays shape-stable (expand-then-project
    FFN would need a two-layer registered module).
    """
    return {
        "n_layers": n_layers,
        "model_config": {
            "transformer": {
                "norm": {"model_name": "LayerNorm", "normalized_shape": hidden_size},
                "attention": {
                    "model_name": "GeneralAttentionBlock",
                    "input_size": hidden_size,
                    "hidden_size": hidden_size,
                    "context_length": seq_len,
                    "heads": num_heads,
                    "causal": False,
                    "multiheaded": True,
                },
                "neural_network": {
                    "model_name": "StandardDense",
                    "in_features": hidden_size,
                    "out_features": hidden_size,
                },
            },
        },
        "device": device,
        "dtype": dtype,
    }


class HierarchicalReasoningModel(DLModule):
    """
    Hierarchical Reasoning Model (HRM) — encoder backbone.

    ``pretrain=True``: MLM logits ``[B, S, vocab_size]`` from ``lm_head``.
    ``pretrain=False``: mean-pooled sentence embeddings ``[B, output_embed_dim]``.
    For classification, wrap with :class:`HRMClassifierWrapper`.
    """

    def __init__(self, config: Union[Dict, HRMConfig], **kwargs):
        kwargs.pop("n_classes", None)
        super().__init__(**kwargs)
        if isinstance(config, dict):
            config = HRMConfig(**config)
        self.hrm_config = config

        # Load tokenizer once from disk (checkpoints/hrm/tokenizer or THESIS_HRM_TOKENIZER_DIR)
        # before building lm_head so vocab_size matches; avoid Hugging Face Hub when local copy exists.
        _hrm_tok = None
        _tok_name = getattr(self.hrm_config, "tokenizer_name", None)
        if _tok_name:
            try:
                from transformers import AutoTokenizer

                src, local_only = _resolve_hrm_tokenizer_source(_tok_name)
                if src:
                    _hrm_tok = AutoTokenizer.from_pretrained(src, local_files_only=local_only)
                    vs = int(getattr(_hrm_tok, "vocab_size", self.hrm_config.vocab_size))
                    self.hrm_config.vocab_size = vs
            except Exception as e:
                _hrm_tok = None
                print(f"Warning: Failed to load tokenizer {_tok_name}: {e}")

        # 1. Fetch the Model Builder Factory
        try:
            from ..transformers.models import get_models
        except ImportError:
            raise ImportError(
                "HierarchicalReasoningModel requires access to `Code.models.deep_learning.transformers`. "
                "Ensure it is installed and accessible."
            )
        
        # 2. Instantiate Inner Models dynamically
        h_model_class = get_models(self.hrm_config.h_level_model)
        l_model_class = get_models(self.hrm_config.l_level_model)

        mk_all = dict(self.hrm_config.model_kwargs)
        mk_enc = dict(mk_all)
        n_layers = int(mk_enc.pop("num_layers", mk_enc.pop("n_layers", 2)))
        num_heads = int(mk_enc.pop("num_heads", 12))
        device_kw = str(kwargs.get("device", "cpu"))
        dtype_kw = kwargs.get("dtype", torch.float32)
        if not isinstance(dtype_kw, torch.dtype):
            dtype_kw = torch.float32

        enc_kw = _inner_encoder_lm_kwargs(
            self.hrm_config.hidden_size,
            n_layers,
            num_heads,
            int(self.hrm_config.seq_len),
            device_kw,
            dtype_kw,
        )
        enc_kw.update(mk_enc)

        if self.hrm_config.h_level_model.lower() == "encoderlm":
            self.H_level = h_model_class(**enc_kw)
        else:
            self.H_level = h_model_class(
                hidden_size=self.hrm_config.hidden_size, **mk_all
            )

        if self.hrm_config.l_level_model.lower() == "encoderlm":
            self.L_level = l_model_class(**enc_kw)
        else:
            self.L_level = l_model_class(
                hidden_size=self.hrm_config.hidden_size, **mk_all
            )
        
        # 3. Initialize Recurrent Persistent States (Learnable Parameters) 
        # Shape: [hidden_size] matched dynamically across batches/seqs later
        self.H_init = nn.Parameter(torch.randn(self.hrm_config.hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(self.hrm_config.hidden_size) * 0.02)
        
        d_out = int(self.hrm_config.output_embed_dim)
        self.output_projection = nn.Linear(self.hrm_config.hidden_size, d_out)
        self.lm_head = nn.Linear(self.hrm_config.hidden_size, self.hrm_config.vocab_size)

        # 5. Text embeddings (tokenizer loaded above)
        self.tokenizer = _hrm_tok
        self.text_embeddings = None
        if _hrm_tok is not None:
            self.text_embeddings = nn.Embedding(self.hrm_config.vocab_size, self.hrm_config.hidden_size)
        
    def _get_initial_carry(self, batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device) -> HRMInnerCarry:
        """Helper to construct the initial hidden states z_H and z_L."""
        # Expand the learnable init vector across batch and sequence dimensions
        z_H = self.H_init.view(1, 1, -1).expand(batch_size, seq_len, -1).to(dtype=dtype, device=device)
        z_L = self.L_init.view(1, 1, -1).expand(batch_size, seq_len, -1).to(dtype=dtype, device=device)
        return HRMInnerCarry(z_H=z_H, z_L=z_L)

    def forward(self, x: Union[torch.Tensor, list, tuple], pretrain: bool = False, **kwargs) -> torch.Tensor:
        """
        ``pretrain=True``: MLM logits ``[B, S, V]``.
        ``pretrain=False``: sentence embeddings ``[B, output_embed_dim]``.
        """
        if self.tokenizer is not None and isinstance(x, (list, tuple)):
            device = next(self.parameters()).device
            tokens = self.tokenizer(list(x), padding=True, truncation=True, max_length=self.hrm_config.seq_len, return_tensors='pt')
            x = tokens['input_ids'].to(device)
            
        if isinstance(x, torch.Tensor) and x.dtype in [torch.long, torch.int]:
            device = next(self.parameters()).device
            x = x.to(device)
            if self.text_embeddings is not None:
                x = self.text_embeddings(x)

        if x.dim() == 2: # [batch, hidden_size] -> [batch, 1, hidden_size]
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.shape
        
        # 1. Initialize Recurrent Carry States
        carry = self._get_initial_carry(batch_size, seq_len, x.dtype, x.device)
        z_H, z_L = carry.z_H, carry.z_L
        
        # 2. Deep Recurrent Loop (Following sapientinc mechanics)
        for _H_step in range(self.hrm_config.H_cycles):
            for _L_step in range(self.hrm_config.L_cycles):
                # Only execute L_level if it's not the absolute final step 
                # (which is usually processed with gradients in full implementations, 
                #  but for simplicity here we just execute it)
                
                # Combine L state with (H state + Input injection)
                l_input = z_L + z_H + x 
                
                # Model execution logic. Models usually return raw logic or logits. 
                # We assume the models function as standard sequence-to-sequence map layers here returning [b, s, h]
                z_L = self.L_level(l_input)
                
                # Unwrap if the transformer output is a tuple/dict containing hidden states
                if isinstance(z_L, tuple):
                    z_L = z_L[0]
                elif hasattr(z_L, 'last_hidden_state'):
                    z_L = z_L.last_hidden_state
                    
                if z_L.dim() == 3 and z_L.shape[-1] != l_input.shape[-1] and z_L.shape[1] == l_input.shape[-1]:
                    z_L = z_L.transpose(1, 2)
                    
            if not (_H_step == self.hrm_config.H_cycles - 1):
                # Update H state using L state
                h_input = z_H + z_L
                z_H = self.H_level(h_input)
                
                if isinstance(z_H, tuple):
                    z_H = z_H[0]
                elif hasattr(z_H, 'last_hidden_state'):
                    z_H = z_H.last_hidden_state
                    
                if z_H.dim() == 3 and z_H.shape[-1] != h_input.shape[-1] and z_H.shape[1] == h_input.shape[-1]:
                    z_H = z_H.transpose(1, 2)
                    
        # 3. Final Step for Readout
        h_input = z_H + z_L
        z_H_final = self.H_level(h_input)
        if hasattr(z_H_final, 'last_hidden_state'):
             z_H_final = z_H_final.last_hidden_state
        elif isinstance(z_H_final, tuple):
             z_H_final = z_H_final[0]

        if pretrain:
            return self.lm_head(z_H_final)
        z_e = self.output_projection(z_H_final)
        return z_e.mean(dim=1)


def build_sentiment_mlp_head(in_dim: int, num_classes: int) -> nn.Sequential:
    """100→320→…→K logits (no softmax); use with CrossEntropyLoss."""
    d_in = int(in_dim)
    k = int(num_classes)
    return nn.Sequential(
        nn.Linear(d_in, 320),
        nn.ReLU(inplace=True),
        nn.Linear(320, 640),
        nn.GELU(),
        nn.Linear(640, 1250),
        nn.GELU(),
        nn.Linear(1250, 640),
        nn.ReLU(inplace=True),
        nn.Linear(640, 320),
        nn.GELU(),
        nn.Linear(320, k),
    )


class HRMClassifierWrapper(nn.Module):
    """Encoder-only HRM plus linear or custom head for supervised fine-tuning."""

    def __init__(
        self,
        encoder: HierarchicalReasoningModel,
        n_classes: int,
        head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        d = int(encoder.hrm_config.output_embed_dim)
        if head is not None:
            self.head = head
        else:
            self.head = nn.Linear(d, int(n_classes))

    def forward(self, x, pretrain: bool = False, **kwargs):
        if pretrain:
            return self.encoder(x, pretrain=True, **kwargs)
        return self.head(self.encoder(x, pretrain=False, **kwargs))
