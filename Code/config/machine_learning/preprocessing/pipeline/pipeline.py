"""Config templates for pipeline."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for Pipeline."""
class PipelineConfig(ConfigTemplate):
    model_name = "Pipeline"
    model_path = "Code.models.machine_learning.preprocessing.pipeline.pipeline"

    def __init__(self,
        immutable: bool = True,
        steps: Union[List[MLModule], Tuple[MLModule], MLModule] = None,
        transform_input: Union[List[str], Tuple[str], str] = None,
        memory: Union[str, object] = 'torch',
        verbose: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.steps = steps
        self.transform_input = transform_input
        self.memory = memory
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
