import torch
import torch.nn as nn
from typing import Optional, Union, Any, Tuple, List, Dict, Callable
from .....models.utils import DLModule
import warnings

from ..attention import get_attention
from ..embeddings import get_embeddings
from ..logits_calculation import get_logits_calculation
from ..neural_network import get_neural_network
from ..norm import get_norm
from ..pooling import get_pooling
from ..positional_encoders import get_positional_encoders
from ..token_selection import get_token_selection
from ..tokenizer import get_tokenizer

from .....config.deep_learning import instantiate_model

from .....models.models import Pipeline


def _parse_module_config(
    module_config: Optional[Union[List[Dict], Tuple[Dict], Dict]],
) -> Dict[str, Optional[Dict]]:
    """
    Normalise module_config into a canonical dict with keys:
      "norm1", "norm2", "attention", "dropout1", "dropout2", "neural_network".

    Accepted forms
    --------------
    Dict
        Shorthand keys: ``"norm"`` applies to both norm1 and norm2;
        ``"dropout"`` applies to both dropout1 and dropout2.
        Explicit keys ``"norm1"``, ``"norm2"``, ``"dropout1"``, ``"dropout2"``
        override their respective shorthands.

    List / Tuple of length 4
        Positional order: [norm_cfg, attention_cfg, dropout_cfg, neural_network_cfg].
        ``norm_cfg`` is shared between norm1 and norm2;
        ``dropout_cfg`` is shared between dropout1 and dropout2.

    List / Tuple of length 6
        Fully explicit positional order:
        [norm1_cfg, attention_cfg, dropout1_cfg, norm2_cfg, neural_network_cfg, dropout2_cfg].

    "Same as block" fallback
    ------------------------
    If norm2 is still None after parsing but norm1 has a config, norm2 inherits norm1's config
    (image annotation: "same as block 1"). Same logic for dropout2 ← dropout1 ("same as block 3").
    """
    result: Dict[str, Optional[Dict]] = {
        "norm1": None,
        "norm2": None,
        "attention": None,
        "dropout1": None,
        "dropout2": None,
        "neural_network": None,
    }

    if module_config is None:
        return result

    if isinstance(module_config, dict):
        # Shorthands first
        if "norm" in module_config:
            result["norm1"] = module_config["norm"]
            result["norm2"] = module_config["norm"]
        if "dropout" in module_config:
            result["dropout1"] = module_config["dropout"]
            result["dropout2"] = module_config["dropout"]
        # Explicit keys override shorthands
        for key in ("norm1", "norm2", "attention", "dropout1", "dropout2", "neural_network"):
            if key in module_config:
                result[key] = module_config[key]

    elif isinstance(module_config, (list, tuple)):
        n = len(module_config)
        if n == 4:
            norm_cfg, attn_cfg, drop_cfg, nn_cfg = module_config
            result["norm1"] = norm_cfg
            result["norm2"] = norm_cfg
            result["attention"] = attn_cfg
            result["dropout1"] = drop_cfg
            result["dropout2"] = drop_cfg
            result["neural_network"] = nn_cfg
        elif n == 6:
            # Fully explicit
            result["norm1"] = module_config[0]
            result["attention"] = module_config[1]
            result["dropout1"] = module_config[2]
            result["norm2"] = module_config[3]
            result["neural_network"] = module_config[4]
            result["dropout2"] = module_config[5]
        else:
            warnings.warn(
                f"module_config as list/tuple must have 4 or 6 elements, got {n}. "
                "Ignoring module_config.",
                stacklevel=3,
            )

    # "Same as block" fallback
    if result["norm2"] is None and result["norm1"] is not None:
        result["norm2"] = result["norm1"]
    if result["dropout2"] is None and result["dropout1"] is not None:
        result["dropout2"] = result["dropout1"]

    return result


def _resolve_module(
    explicit_module: Optional[nn.Module],
    cfg: Optional[Dict],
) -> Optional[nn.Module]:
    """
    Resolve a single sub-module with the following priority:

    1. ``explicit_module`` — returned as-is if not None.
    2. ``cfg`` with a ``"model_name"`` key — instantiated via
       ``instantiate_model(model_name, overrides)``.
    3. None — caller is responsible for the fallback (e.g. ``nn.Identity``).
    """
    if explicit_module is not None:
        return explicit_module
    if cfg is not None:
        model_name = cfg.get("model_name")
        if model_name:
            overrides = {k: v for k, v in cfg.items() if k != "model_name"}
            return instantiate_model(model_name, overrides)
    return None


class AddBlock(nn.Module):
    """
    Splits the concatenated input from multiple predecessors into two equal parts
    and adds them element-wise.

    In ``Pipeline`` multiple edges pointing to a single node are concatenated
    along ``dim=-1``; this block reverses that concatenation into a residual sum.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) % 2 != 0:
            return x
        half = x.size(-1) // 2
        return x[..., :half] + x[..., half:]


class GeneralTransformer(DLModule):
    """
    General Transformer Block implemented via ``Pipeline``.

    Pipeline topology (7 functional blocks + 2 residual ADD nodes)::

        IN ──► NORM1 ──► ATTN ──► DROP1 ──► ADD1 ──► NORM2 ──► FF ──► DROP2 ──► ADD2 ──► OUT
         ╰──────────────────────────────────► ADD1 ╯    ╰──────────────────────────────► ADD2 ╯

    Block roles
    -----------
    Block 1 – NORM1  : Norm (Any class from the norm sub-package)
    Block 2 – ATTN   : Attention (Any class from the attention sub-package)
    Block 3 – DROP1  : Dropout (Optional)
    Block 4 – ADD1   : Residual add (LAZY – auto-splits the concatenated skip+path tensors)
    Block 5 – NORM2  : Norm (same class/config as Block 1 by default)
    Block 6 – FF     : Custom Neural Network (Any)
    Block 7 – DROP2  : Dropout (same class/config as Block 3 by default)
               ADD2  : Residual add (LAZY)

    Parameters
    ----------
    norm1_module : nn.Module, optional
        Pre-built norm for Block 1. Takes precedence over ``module_config``.
    attention_module : nn.Module, optional
        Pre-built attention for Block 2. Takes precedence over ``module_config``.
    dropout1_module : nn.Module, optional
        Pre-built dropout for Block 3. Takes precedence over ``module_config``.
    norm2_module : nn.Module, optional
        Pre-built norm for Block 5. Takes precedence over ``module_config``.
        Defaults to a fresh instance from the same config as Block 1.
    neural_network_module : nn.Module, optional
        Pre-built feed-forward network for Block 6. Takes precedence over ``module_config``.
    dropout2_module : nn.Module, optional
        Pre-built dropout for Block 7. Takes precedence over ``module_config``.
        Defaults to a fresh instance from the same config as Block 3.
    module_config : dict | list | tuple, optional
        Config-driven module specification.  Accepted forms:

        *Dict* (recommended)::

            {
                "norm":           {"model_name": "LayerNorm", "normalized_shape": 512},
                "attention":      {"model_name": "MultiHeadAttention", "num_heads": 8},
                "dropout":        {"model_name": "Dropout", "p": 0.1},
                "neural_network": {"model_name": "StandardDense", "in_features": 512},
            }

        Use ``"norm1"``/``"norm2"`` and ``"dropout1"``/``"dropout2"`` keys to configure
        the two norms or dropouts **independently**; the shorthand ``"norm"``/``"dropout"``
        keys share the same config for both.

        *List / Tuple of length 4* – positional:
        ``[norm_cfg, attention_cfg, dropout_cfg, neural_network_cfg]``
        (norm and dropout configs are shared between the two occurrences).

        *List / Tuple of length 6* – fully explicit:
        ``[norm1_cfg, attention_cfg, dropout1_cfg, norm2_cfg, neural_network_cfg, dropout2_cfg]``

        Any module not covered by ``module_config`` falls back to ``nn.Identity``.

    causal : bool
        If True, passes ``causal=True`` to the attention module.
    cross : bool
        If True, passes ``cross=True`` to the attention module.
    """

    def __init__(
        self,
        norm1_module: Optional[nn.Module] = None,
        attention_module: Optional[nn.Module] = None,
        dropout1_module: Optional[nn.Module] = None,
        norm2_module: Optional[nn.Module] = None,
        neural_network_module: Optional[nn.Module] = None,
        dropout2_module: Optional[nn.Module] = None,
        module_config: Optional[Union[List[Dict], Tuple[Dict], Dict]] = None,
        causal: bool = False,
        cross: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.cross = cross

        cfg = _parse_module_config(module_config)

        norm1 = _resolve_module(norm1_module, cfg["norm1"]) or nn.Identity()
        attn = _resolve_module(attention_module, cfg["attention"]) or nn.Identity()
        drop1 = _resolve_module(dropout1_module, cfg["dropout1"]) or nn.Identity()
        norm2 = _resolve_module(norm2_module, cfg["norm2"]) or nn.Identity()
        ff = _resolve_module(neural_network_module, cfg["neural_network"]) or nn.Identity()
        drop2 = _resolve_module(dropout2_module, cfg["dropout2"]) or nn.Identity()

        # Apply causality / cross-attention overrides to the attention module
        if hasattr(attn, "causal"):
            attn.causal = causal
        if hasattr(attn, "cross"):
            attn.cross = cross

        mermaid_flowchart = """
        graph TD
            IN[Input]
            NORM1[Norm]
            ATTN[Attention]
            DROP1[Dropout]
            ADD1[Add]
            NORM2[Norm]
            FF[NeuralNetwork]
            DROP2[Dropout]
            ADD2[Add]
            OUT[Output]

            IN --> NORM1
            NORM1 --> ATTN
            ATTN --> DROP1

            IN --> ADD1
            DROP1 --> ADD1

            ADD1 --> NORM2
            NORM2 --> FF
            FF --> DROP2

            ADD1 --> ADD2
            DROP2 --> ADD2

            ADD2 --> OUT
        """

        modules = {
            "IN": nn.Identity(),
            "NORM1": norm1,
            "ATTN": attn,
            "DROP1": drop1,
            "ADD1": AddBlock(),
            "NORM2": norm2,
            "FF": ff,
            "DROP2": drop2,
            "ADD2": AddBlock(),
            "OUT": nn.Identity(),
        }

        self.pipeline = Pipeline(
            mermaid_flowchart=mermaid_flowchart,
            modules=modules,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Hardcoded explicit topology matching the original flowchart intention
        # without relying on Pipeline's sequential (non-branching) iterator.
        
        blocks = self.pipeline.blocks
        
        # Block 1 - Attention
        residual = x
        out = blocks["NORM1"](x)
        out = blocks["ATTN"](out)
        if isinstance(out, tuple):
            out = out[0]
        out = blocks["DROP1"](out)
        out = out + residual
        
        # Block 2 - Feed Forward
        residual = out
        out = blocks["NORM2"](out)
        out = blocks["FF"](out)
        out = blocks["DROP2"](out)
        out = out + residual
        
        return out
