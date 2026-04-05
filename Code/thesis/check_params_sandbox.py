import os
import json
import torch
import importlib
import inspect
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)

def get_class_from_name(class_name):
    base_dir = os.path.join(ROOT, "Code", "models")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                rel_path = os.path.relpath(os.path.join(root, file), ROOT)
                module_path = rel_path.replace(os.sep, ".").replace(".py", "")
                try:
                    mod = importlib.import_module(module_path)
                    if hasattr(mod, class_name):
                        return getattr(mod, class_name)
                except Exception:
                    pass
    return None

def instantiate_model(cls, kwargs):
    # Fix @torch strings
    for k, v in list(kwargs.items()):
        if isinstance(v, str) and v.startswith("@torch."):
            kwargs[k] = getattr(torch, v.split(".")[1])
            
    try:
        sig = inspect.signature(cls.__init__)
        valid_keys = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD or p.kind == p.KEYWORD_ONLY]
        
        has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        if has_varkw:
            filtered_kwargs = kwargs
        else:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            
        return cls(**filtered_kwargs)
    except Exception as e:
        print(f"Instantiation via inspect failed: {e}. Trying full kwargs.")
        return cls(**kwargs)

config_dir = os.path.join(ROOT, "Code", "thesis", "config")
for root, _, files in os.walk(config_dir):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                config = json.load(f)
            
            for class_name, params in config.items():
                print(f"--- {file} ({class_name}) ---")
                cls = get_class_from_name(class_name)
                if not cls:
                    print("Class not found")
                    continue
                try:
                    model = instantiate_model(cls, params)
                    if hasattr(model, "parameters"):
                        num_params = sum(p.numel() for p in model.parameters())
                        print(f"Success! Params: {num_params:,}")
                    else:
                        print("Success! (Not a typical PyTorch nn.Module with .parameters())")
                except Exception as e:
                    print(f"Failed: {e}")
