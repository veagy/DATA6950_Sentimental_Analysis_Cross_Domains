"""Config templates for feature_extraction."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from ....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for DictVectorizer."""
class DictVectorizerConfig(ConfigTemplate):
    model_name = "DictVectorizer"
    model_path = "Code.models.machine_learning.feature_extraction.feature_extractio"

    def __init__(self,
        immutable: bool = True,
        separator: str = '=',
        sparse: bool = True,
        sort: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        self.device = device
        self.dtype = dtype
