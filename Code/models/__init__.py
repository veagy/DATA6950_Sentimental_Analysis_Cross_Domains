"""
Unified models package: DL, ML, and transformer architectures.

Exports every public class and function (non-underscore-prefixed) from all
sub-packages: utils, deep_learning (activations, rnn, cnn, dbn, ffnn, llm,
transformers), and machine_learning. Includes the universal get_model() factory.
"""

from ..models.utils import (
    DLModule,
    DLModuleWrapper,
    MLModule,
    MLRegressor,
    MLClassifier,
    MLCluster,
    MLTransform,
    ActFuncWrapper,
    ActFuncUtils,
    Forward_hook,
    Backward_hook,
)
from ..models.deep_learning import (
    NormalizationLayer,
    PoolingLayer,
    DropoutLayer,
    ConvolutionLayer,
    PaddingLayer,
    TransformerLayer,
    DLModelLayers,
    SoftDTWBatch,
    SoftDTWSimilarity,
    SoftDTWMatrix,
)
try:
    from ..models.deep_learning.activations import *  # noqa: F401, F403
    from ..models.deep_learning.activations import __all__ as _activations_all
except ImportError:  # Adaptive submodule files may be absent in minimal checkouts
    _activations_all = []  # type: ignore[misc]
from ..models.deep_learning.rnn import *  # noqa: F401, F403
from ..models.deep_learning.rnn import __all__ as _rnn_all
from ..models.deep_learning.cnn import *  # noqa: F401, F403
from ..models.deep_learning.cnn import __all__ as _cnn_all
try:
    from ..models.deep_learning.dbn import *  # noqa: F401, F403
    from ..models.deep_learning.dbn import __all__ as _dbn_all
except ImportError:
    _dbn_all = []  # type: ignore[misc]
try:
    from ..models.deep_learning.ffnn import *  # noqa: F401, F403
    from ..models.deep_learning.ffnn import __all__ as _ffnn_all
except ImportError:
    _ffnn_all = []  # type: ignore[misc]
try:
    from ..models.deep_learning.llm import LLMModule
    _llm_all = ["LLMModule"]
except (ImportError, ModuleNotFoundError):
    LLMModule = None  # type: ignore[misc, assignment]
    _llm_all = []
from ..models.deep_learning.transformers.attention import *  # noqa: F401, F403
from ..models.deep_learning.transformers.attention import __all__ as _attn_all
from ..models.deep_learning.transformers.embeddings import *  # noqa: F401, F403
from ..models.deep_learning.transformers.embeddings import __all__ as _emb_all
from ..models.deep_learning.transformers.norm import *  # noqa: F401, F403
from ..models.deep_learning.transformers.norm import __all__ as _norm_all
from ..models.deep_learning.transformers.pooling import *  # noqa: F401, F403
from ..models.deep_learning.transformers.pooling import __all__ as _pool_all
from ..models.deep_learning.transformers.positional_encoders import *  # noqa: F401, F403
from ..models.deep_learning.transformers.positional_encoders import __all__ as _pe_all
from ..models.deep_learning.transformers.logits_calculation import *  # noqa: F401, F403
from ..models.deep_learning.transformers.logits_calculation import __all__ as _lc_all
from ..models.deep_learning.transformers.token_selection import *  # noqa: F401, F403
from ..models.deep_learning.transformers.token_selection import __all__ as _ts_all
from ..models.deep_learning.transformers.tokenizer import *  # noqa: F401, F403
from ..models.deep_learning.transformers.tokenizer import __all__ as _tok_all
from ..models.deep_learning.transformers.neural_network import *  # noqa: F401, F403
from ..models.deep_learning.transformers.neural_network import __all__ as _nn_all
try:
    from ..models.deep_learning.transformers.models import *  # noqa: F401, F403
    from ..models.deep_learning.transformers.models import __all__ as _tfm_all
except (ImportError, ModuleNotFoundError):
    _tfm_all = []
from ..models.machine_learning import *  # noqa: F401, F403
from ..models.machine_learning import __all__ as _ml_all


def get_model(name: str, *args, **kwargs):
    """Universal model factory: resolves any class across ALL src/models/ sub-packages.

    Resolution order:
      1. transformers.models.get_models() (fast in-memory, covers transformer architectures)
      2. model_registry.get_model_module_path() + importlib (covers everything else)

    When *args or **kwargs are provided, instantiates and returns the model
    (wrapped with DLModuleWrapper if not already a DLModule). Otherwise returns
    the class.

    Parameters
    ----------
    name : str
        Class name (case-insensitive for transformer models).
    *args, **kwargs
        Optional constructor arguments. When provided, the class is instantiated.

    Returns
    -------
    type | DLModule
        The model class when called with no extra args, or an instantiated
        model (possibly wrapped) when args/kwargs are supplied.

    Raises
    ------
    KeyError
        When the name is not found in transformers or the model registry.
    """
    cls = None
    try:
        from ..models.deep_learning.transformers.models import get_models
        cls = get_models(name)
    except (KeyError, ImportError, ModuleNotFoundError):
        pass
    if cls is None:
        from ..config.deep_learning.model_registry import get_model_module_path
        import importlib
        module_path = get_model_module_path(name)
        m = importlib.import_module(module_path)
        cls = getattr(m, name)
    if args or kwargs:
        inst = cls(*args, **kwargs)
        if not isinstance(inst, DLModule):
            inst = DLModuleWrapper.wrap(inst)
        return inst
    return cls


_utils_all = [
    "DLModule",
    "DLModuleWrapper",
    "MLModule",
    "MLRegressor",
    "MLClassifier",
    "MLCluster",
    "MLTransform",
    "ActFuncWrapper",
    "ActFuncUtils",
    "Forward_hook",
    "Backward_hook",
]
_dl_all = [
    "NormalizationLayer",
    "PoolingLayer",
    "DropoutLayer",
    "ConvolutionLayer",
    "PaddingLayer",
    "TransformerLayer",
    "DLModelLayers",
    "SoftDTWBatch",
    "SoftDTWSimilarity",
    "SoftDTWMatrix",
]

__all__ = (
    _utils_all
    + _dl_all
    + list(_activations_all)
    + list(_rnn_all)
    + list(_cnn_all)
    + list(_dbn_all)
    + list(_ffnn_all)
    + _llm_all
    + list(_attn_all)
    + list(_emb_all)
    + list(_norm_all)
    + list(_pool_all)
    + list(_pe_all)
    + list(_lc_all)
    + list(_ts_all)
    + list(_tok_all)
    + list(_nn_all)
    + list(_tfm_all)
    + list(_ml_all)
    + ["get_model"]
)
