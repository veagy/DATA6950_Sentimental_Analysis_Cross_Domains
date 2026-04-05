import joblib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Union, Optional

import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file, load_model
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


class SentinelSerializer:
    """
    Unified central serialization engine for the Sentinel system.
    Supports PyTorch native (.pt, .pth), Safetensors (.safetensors), ONNX (.onnx), 
    TorchScript (.ts), and joblib/pickle for classical ML components.
    All disk writes are guaranteed atomic using tempfiles.
    """

    @staticmethod
    def _is_safetensor_type(ext: str, save_type: str) -> bool:
        return save_type.lower() == "safetensors" or ext in {".safetensors", ".st"}

    @staticmethod
    def _is_onnx_type(ext: str, save_type: str) -> bool:
        return save_type.lower() == "onnx" or ext == ".onnx"

    @staticmethod
    def _is_torchscript_type(ext: str, save_type: str) -> bool:
        return save_type.lower() in ("ts", "jit") or ext in {".ts", ".jit"}
        
    @staticmethod
    def save(model: nn.Module, filepath: Union[str, Path], save_type: str = "pt", input_shape: Optional[tuple] = None) -> None:
        """
        Saves a PyTorch nn.Module using the prescribed format (derived from extension or override).
        Writes atomically via temporary files.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        ext = filepath.suffix.lower()

        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix="sentinel_save_")
        os.close(fd)

        try:
            if SentinelSerializer._is_safetensor_type(ext, save_type):
                if not _SAFETENSORS_AVAILABLE:
                    raise ImportError("safetensors library is not installed.")
                save_file(model.state_dict(), tmp_path)
                
            elif SentinelSerializer._is_onnx_type(ext, save_type):
                if input_shape is None:
                    raise ValueError("ONNX export requires an input_shape parameters to trace the graph.")
                device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
                dummy_input = torch.randn(*input_shape, device=device)
                torch.onnx.export(model, dummy_input, tmp_path, export_params=True)
                
            elif SentinelSerializer._is_torchscript_type(ext, save_type):
                scripted = torch.jit.script(model)
                scripted.save(tmp_path)
                
            else:
                # Default PyTorch (.pt, .pth)
                torch.save(model.state_dict(), tmp_path)

            os.replace(tmp_path, filepath)
            
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Failed to serialize model to {filepath}: {e}")

    @staticmethod
    def load(model: nn.Module, filepath: Union[str, Path], device: Union[str, torch.device] = "cpu", strict: bool = False) -> None:
        """
        Attempts to load weights from disk into the provided structure. Supports format auto-detection.
        Note: ONNX and TorchScript cannot easily route back cleanly into a native PyTorch model parameter map dict,
        so this primarily targets .pt and .safetensors.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

        ext = filepath.suffix.lower()
        
        if SentinelSerializer._is_safetensor_type(ext, ext.replace(".", "")):
            if not _SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors library is not installed.")
            # Safetensors loads directly into the module.
            load_model(model, str(filepath))
            model.to(device)
            return

        elif SentinelSerializer._is_torchscript_type(ext, ext.replace(".", "")):
            raise ValueError("TorchScript models (.ts) should be loaded directly via torch.jit.load(), not projected back onto arbitrary modules.")
            
        elif SentinelSerializer._is_onnx_type(ext, ext.replace(".", "")):
            raise ValueError("ONNX models (.onnx) cannot be loaded back into PyTorch nn.Modules natively. Use an ONNX inference runtime.")

        else:
            # Default PyTorch (.pt, .pth)
            state = torch.load(str(filepath), map_location=device)
            # Handle if the container saved the entire module rather than the state dict
            if isinstance(state, nn.Module):
                state = state.state_dict()
            model.load_state_dict(state, strict=strict)
            model.to(device)

    @staticmethod
    def save_object(obj_dict: Dict[str, Any], filepath: Union[str, Path], method: str = 'joblib') -> None:
        """Saves a python dictionary/object (legacy ML preprocessors)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix="obj_save_")
        os.close(fd)
        
        try:
            if method == 'joblib':
                joblib.dump(obj_dict, tmp_path)
            elif method == 'pickle':
                with open(tmp_path, 'wb') as f:
                    pickle.dump(obj_dict, f)
            else:
                raise ValueError(f"Unsupported serialization method: {method}")
                
            os.replace(tmp_path, filepath)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

    @staticmethod
    def load_object(filepath: Union[str, Path], method: str = 'joblib') -> Dict[str, Any]:
        """Loads a python dictionary/object."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        if method == 'joblib':
            return joblib.load(str(filepath))
        elif method == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported serialization method: {method}")

# Legacy API retention
save_preprocessor = SentinelSerializer.save_object
load_preprocessor = SentinelSerializer.load_object
