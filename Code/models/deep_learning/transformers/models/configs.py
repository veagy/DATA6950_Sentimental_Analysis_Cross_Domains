from __future__ import annotations
"""
Transformer model configuration dataclasses.

Aligns with HuggingFace PreTrainedConfig parameters for flexible architecture
customization. Models accept config=None and **kwargs; when config is provided
it takes precedence, else kwargs are used with defaults.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


def _merge_config_kwargs(
    config: Optional[Any],
    kwargs: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge config attributes, kwargs, and defaults. kwargs override config, config overrides defaults."""
    result = dict(defaults)
    if config is not None:
        for k, v in defaults.items():
            if hasattr(config, k):
                result[k] = getattr(config, k)
    for k, v in kwargs.items():
        if k in result:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Common (PreTrainedConfig base)
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Base config with common parameters for text/encoder-decoder models."""

    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    use_cache: bool = True


# ---------------------------------------------------------------------------
# LLaMA / Mistral / Falcon / Gemma / Phi3 / Qwen2 (decoder-only)
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig(TransformerConfig):
    """LLaMA-style decoder config. Aliases: embed_dim=hidden_size, ffn_dim=intermediate_size."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 2048
    hidden_act: str = "silu"
    num_key_value_heads: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class MistralConfig(LlamaConfig):
    """Mistral: sliding-window + GQA."""

    window_size: int = 4096
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768


@dataclass
class FalconConfig(LlamaConfig):
    """Falcon: Multi-Query Attention + parallel attn+FFN."""

    num_key_value_heads: int = 1
    parallel_attn: bool = True


@dataclass
class GemmaConfig(LlamaConfig):
    """Gemma: GQA + logit soft-capping."""

    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 8192


@dataclass
class Phi3Config(LlamaConfig):
    """Phi-3: tiny but dense, long-context RoPE scaling."""

    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096


@dataclass
class Qwen2Config(LlamaConfig):
    """Qwen2: dual-chunk attention + dynamic NTK RoPE."""

    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 32768


# ---------------------------------------------------------------------------
# RoBERTa / BERT / ALBERT / DeBERTa (encoder-only)
# ---------------------------------------------------------------------------

@dataclass
class BertConfig(TransformerConfig):
    """BERT/RoBERTa encoder config."""

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


@dataclass
class RobertaConfig(BertConfig):
    """RoBERTa config."""

    vocab_size: int = 50265


@dataclass
class AlbertConfig(BertConfig):
    """ALBERT: embedding_size, inner_group_num."""

    embedding_size: int = 128
    inner_group_num: int = 1


@dataclass
class DebertaConfig(BertConfig):
    """DeBERTa: relative attention."""

    relative_attention: bool = True
    max_relative_positions: int = 512


# ---------------------------------------------------------------------------
# T5 / BART (encoder-decoder)
# ---------------------------------------------------------------------------

@dataclass
class T5Config(TransformerConfig):
    """T5 encoder-decoder config. Uses d_model, d_ff, d_kv."""

    vocab_size: int = 32128
    hidden_size: int = 512  # d_model
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048  # d_ff
    d_kv: int = 64
    num_decoder_layers: Optional[int] = None
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "relu"
    initializer_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.num_decoder_layers is None:
            self.num_decoder_layers = self.num_hidden_layers


@dataclass
class BARTConfig(T5Config):
    """BART encoder-decoder config."""

    vocab_size: int = 50265
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 1024


@dataclass
class LongT5Config(T5Config):
    """LongT5: TGlobal attention with block_len."""

    block_len: int = 16


# ---------------------------------------------------------------------------
# ViT / DeiT / BEiT / MAE (vision)
# ---------------------------------------------------------------------------

@dataclass
class ViTConfig(TransformerConfig):
    """Vision Transformer config."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    encoder_stride: int = 16
    pooler_output_size: Optional[int] = None
    pooler_act: str = "tanh"

    def __post_init__(self) -> None:
        if self.pooler_output_size is None:
            self.pooler_output_size = self.hidden_size


@dataclass
class SwinConfig(ViTConfig):
    """Swin Transformer: hierarchical, shifted windows."""

    window_size: int = 7
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1


# ---------------------------------------------------------------------------
# DETR (detection)
# ---------------------------------------------------------------------------

@dataclass
class DetrConfig(TransformerConfig):
    """DETR detection transformer config."""

    num_queries: int = 100
    d_model: int = 256
    encoder_layers: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8
    position_embedding_type: str = "sine"
    auxiliary_loss: bool = False
    num_channels: int = 3


# ---------------------------------------------------------------------------
# Mamba / S4 / S5 (SSM)
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig(TransformerConfig):
    """Mamba SSM config."""

    vocab_size: int = 50280
    hidden_size: int = 768
    num_hidden_layers: int = 32
    state_size: int = 16
    conv_kernel: int = 4
    expand: int = 2
    layer_norm_epsilon: float = 1e-5
    time_step_rank: Union[int, str] = "auto"
    time_step_scale: float = 1.0
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    use_bias: bool = False
    use_conv_bias: bool = True
    residual_in_fp32: bool = True


# ---------------------------------------------------------------------------
# Wav2Vec2 (audio)
# ---------------------------------------------------------------------------

@dataclass
class CLIPConfig(TransformerConfig):
    """CLIP multimodal config (flattened vision + text)."""

    vision_hidden_size: int = 768
    vision_num_hidden_layers: int = 12
    vision_num_attention_heads: int = 12
    vision_num_channels: int = 3
    vision_patch_size: int = 16
    vision_image_size: int = 224
    text_hidden_size: int = 512
    text_num_hidden_layers: int = 12
    text_num_attention_heads: int = 8
    text_max_position_embeddings: int = 77
    projection_dim: int = 512


@dataclass
class Wav2Vec2Config(TransformerConfig):
    """Wav2Vec2 audio config."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    conv_dim: Tuple[int, ...] = (512,) * 7
    conv_stride: Tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16


# ---------------------------------------------------------------------------
# Longformer / BigBird (long-context)
# ---------------------------------------------------------------------------

@dataclass
class LongformerConfig(BertConfig):
    """Longformer: attention_window, global attn."""

    attention_window: int = 512
    sep_token_id: int = 2


@dataclass
class BigBirdConfig(BertConfig):
    """BigBird: block size, num_random_blocks."""

    block_size: int = 64
    num_random_blocks: int = 3


# ---------------------------------------------------------------------------
# ESM2 / ProteinLM (scientific)
# ---------------------------------------------------------------------------

@dataclass
class ESM2Config(BertConfig):
    """ESM2 protein language model."""

    vocab_size: int = 33  # amino acids + special
    max_position_embeddings: int = 1022


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class ColBERTConfig(BertConfig):
    """ColBERT retrieval config."""

    max_length: int = 512
    similarity_metric: str = "cosine"
    compression_dim: int = 128


@dataclass
class SASRecConfig(TransformerConfig):
    """SASRec / BERT4Rec sequential recommendation."""

    n_items: int = 10000
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    max_position_embeddings: int = 200
    inner_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.inner_dim is None:
            self.inner_dim = self.hidden_size * 4


# ---------------------------------------------------------------------------
# Generative (DiT, SiT)
# ---------------------------------------------------------------------------

@dataclass
class DiTConfig(TransformerConfig):
    """Diffusion Transformer config."""

    in_channels: int = 4
    patch_size: int = 2
    num_attention_heads: int = 16
    mlp_ratio: float = 4.0
    block_out_channels: Optional[Tuple[int, ...]] = None
    time_embed_dim: Optional[int] = None
    class_embed_type: str = "timestep"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclass
class GraphormerConfig(TransformerConfig):
    """Graphormer config."""

    d_node: int = 9
    d_edge: int = 3
    hidden_size: int = 64
    num_attention_heads: int = 4
    max_nodes: int = 512
    max_edges: int = 1024


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesTransformerConfig(TransformerConfig):
    """Time series / tabular transformer."""

    n_features: int = 64
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    patch_size: int = 16
    forecast_horizon: int = 24
    enc_in: Optional[int] = None
    dec_in: Optional[int] = None
    c_out: Optional[int] = None

    def __post_init__(self) -> None:
        if self.enc_in is None:
            self.enc_in = self.n_features
        if self.dec_in is None:
            self.dec_in = self.n_features
        if self.c_out is None:
            self.c_out = self.n_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert config dataclass to dict for model_config passthrough."""
    if config is None:
        return {}
    if hasattr(config, "__dataclass_fields__"):
        return {k: getattr(config, k) for k in config.__dataclass_fields__}
    return {}


def resolve_decoder_config(
    config: Optional[LlamaConfig],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Resolve LLaMA-style decoder params from config or kwargs. Returns dict with embed_dim, n_heads, etc."""
    defaults = {
        "vocab_size": 32000,
        "embed_dim": 4096,
        "n_heads": 32,
        "n_kv_heads": 32,
        "ffn_dim": 11008,
        "n_layers": 32,
        "max_len": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "attention_bias": False,
        "mlp_bias": False,
        "window_size": None,
    }
    if config is not None:
        defaults["vocab_size"] = config.vocab_size
        defaults["embed_dim"] = config.hidden_size
        defaults["n_heads"] = config.num_attention_heads
        defaults["n_kv_heads"] = config.num_key_value_heads or config.num_attention_heads
        defaults["ffn_dim"] = config.intermediate_size
        defaults["n_layers"] = config.num_hidden_layers
        defaults["max_len"] = config.max_position_embeddings
        defaults["rms_norm_eps"] = getattr(config, "rms_norm_eps", 1e-6)
        defaults["rope_theta"] = getattr(config, "rope_theta", 10000.0)
        defaults["attention_dropout"] = getattr(config, "attention_dropout", 0.0)
        defaults["attention_bias"] = getattr(config, "attention_bias", False)
        defaults["mlp_bias"] = getattr(config, "mlp_bias", False)
        defaults["window_size"] = getattr(config, "window_size", None)
    # Alias kwargs
    aliases = {
        "hidden_size": "embed_dim",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "intermediate_size": "ffn_dim",
        "max_position_embeddings": "max_len",
    }
    for hf_name, our_name in aliases.items():
        if hf_name in kwargs:
            kwargs[our_name] = kwargs.pop(hf_name)
    for k, v in kwargs.items():
        if k in defaults:
            defaults[k] = v
    return defaults
