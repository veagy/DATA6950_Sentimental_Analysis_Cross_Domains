from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file


def save_safetensors(state_dict: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu[k] = v.detach().cpu().contiguous()
    if not cpu:
        cpu["_empty"] = torch.zeros(1, dtype=torch.float32)
    save_file(cpu, str(path))


def load_safetensors_state(
    path: str | Path, map_location: Optional[torch.device | str] = None
) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = load_file(str(path), device="cpu")
    if map_location is not None:
        return {k: v.to(map_location) for k, v in state.items()}
    return dict(state)
