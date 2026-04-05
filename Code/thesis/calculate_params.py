import os
import sys
import json
import importlib
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "models"))

CONFIG_DIR = PROJECT_ROOT / "Code" / "thesis" / "config"
DL_REGISTRY_PATH = PROJECT_ROOT / "Code" / "config" / "deep_learning" / "model_config_registry.json"

if not DL_REGISTRY_PATH.exists():
    print(f"FATAL VERIFICATION ERROR: Deep Learning registry missing at {DL_REGISTRY_PATH}")
    sys.exit(1)

with open(DL_REGISTRY_PATH, 'r') as f:
    dl_registry = json.load(f)

def dynamic_import(class_name):
    # Retrieve nested path (e.g. 'cnn.models.models' or 'hrm.models')
    module_suffix = dl_registry.get(class_name)
    if not module_suffix:
        return None
    
    # Try importing under Code.models.deep_learning
    full_module_path = f"Code.models.deep_learning.{module_suffix}"
    try:
        mod = importlib.import_module(full_module_path)
        return getattr(mod, class_name)
    except Exception as e:
        # Fallback without the .models structure since some values are e.g. "cnn.models"
        try:
            mod = importlib.import_module(f"Code.models.deep_learning.{module_suffix}.models")
            return getattr(mod, class_name)
        except Exception as e2:
            return None

def verify_parameters():
    print("="*80)
    print("           STRICT MATHEMATICAL .NUMEL() VERIFICATION           ")
    print("="*80)
    print(f"{'Folder':<15} | {'Configuration File':<25} | {'Param Count':<25}")
    print("-" * 80)
    
    total_verified = 0
    total_failed = 0
    
    # Iterate dynamically through all thesis configurations
    for root, _, files in os.walk(CONFIG_DIR):
        for file in files:
            if not file.endswith(".json"): continue
            
            filepath = Path(root) / file
            folder_name = filepath.parent.parent.name
            
            with open(filepath, 'r') as f:
                config_data = json.load(f)
                
            if not config_data: continue
            architecture_key = list(config_data.keys())[0]
            params = config_data[architecture_key]
            
            if folder_name in ["ml", "ensemble"]:
                print(f"{folder_name:<15} | {file:<25} | {'Statistical Model':<25}")
                continue
                
            ArchClass = dynamic_import(architecture_key)
            if not ArchClass:
                print(f"{folder_name:<15} | {file:<25} | FAILED: Constructor Not Found")
                total_failed += 1
                continue
                
            clean_params = dict(params)
            clean_params.pop("data_source", None)
            clean_params.pop("source", None)
            
            try:
                # Direct PyTorch Instantiation utilizing the mapped dictionary
                model = ArchClass(**clean_params)
                
                # Mathematical Proof execution
                total_params = sum(p.numel() for p in model.parameters())
                print(f"{folder_name:<15} | {file:<25} | {total_params:<25,}")
                total_verified += 1
                
            except Exception as e:
                print(f"{folder_name:<15} | {file:<25} | INSTANTIATION CRASH: {str(e)[:25]}")
                total_failed += 1
                
    print("="*80)
    print(f"Verified {total_verified} Neural Network graphs successfully. {total_failed} instantiations threw positional definition errors.")
    print("="*80)

if __name__ == "__main__":
    verify_parameters()
