"""Config templates for image."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for PatchExtractor."""
class PatchExtractorConfig(ConfigTemplate):
    model_name = "PatchExtractor"
    model_path = "Code.models.machine_learning.feature_extraction.image.image"

    def __init__(self,
        immutable: bool = True,
        patch_size: torch.Tensor = None,
        max_patches: Union[int, float] = None,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
