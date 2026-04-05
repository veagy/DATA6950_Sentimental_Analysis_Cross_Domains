"""
MoEPipelineBuilder: Programmatic construction and configuration saving for MoE structures.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from pathlib import Path

from ..models import Pipeline
from .parallel import ParallelMoE
from .sequential import SequentialMoE

PATH_USER_CONFIGS_MODELS = Path("./configs/models")


class MoEPipelineBuilder:
    """
    Factory class to instantiate MoE PyTorch Modules, generate their Mermaid definitions,
    dump them to the pipeline config directory, and wrap them in the system's interactive Pipeline.
    """
    
    @staticmethod
    def _save_mermaid(name: str, structure: str) -> Path:
        """Saves mermaid definition to local project definitions."""
        PATH_USER_CONFIGS_MODELS.mkdir(parents=True, exist_ok=True)
        file_path = PATH_USER_CONFIGS_MODELS / f"{name}.mmd"
        file_path.write_text(structure)
        return file_path

    @classmethod
    def create_parallel_moe_pipeline(
        cls,
        name: str,
        experts: Dict[str, nn.Module],
        in_features: int,
        hrm: Optional[nn.Module] = None,
        gating_hidden_dim: Optional[int] = None,
        pipeline_kwargs: Optional[Dict] = None,
    ) -> Pipeline:
        """
        Constructs a ParallelMoE, maps it to Mermaid config, and returns initialized Pipeline.
        """
        moe_module = ParallelMoE(
            experts=nn.ModuleDict(experts),
            in_features=in_features,
            hrm=hrm,
            gating_hidden_dim=gating_hidden_dim
        )
        
        # Build simplified Mermaid Architecture definition for execution
        expert_names = list(experts.keys())
        mmd_lines = [
            "graph TD",
            "    %% Conceptual Structure (Parallel MoE):",
        ]
        
        if hrm is not None:
            mmd_lines.append("    %% HRM --> GATING")
            for expr in expert_names:
                mmd_lines.append(f"    %% HRM --> {expr}")
        else:
            mmd_lines.append("    %% IN --> GATING")
            for expr in expert_names:
                mmd_lines.append(f"    %% IN --> {expr}")
                
        mmd_lines.extend([
            "    %% execution DAG:",
            "    IN[Input] --> MOE_WRAPPER",
            "    MOE_WRAPPER --> OUT[Output]"
        ])
        
        mermaid_text = "\n".join(mmd_lines)
        cls._save_mermaid(name, mermaid_text)
        
        kwargs = pipeline_kwargs or {}
        return Pipeline(
            template_name=name,
            modules={"MOE_WRAPPER": moe_module},
            **kwargs
        )

    @classmethod
    def create_sequential_moe_pipeline(
        cls,
        name: str,
        experts: Union[List[nn.Module], nn.ModuleList],
        thresholds: Union[float, List[float]],
        pipeline_kwargs: Optional[Dict] = None,
    ) -> Pipeline:
        """
        Constructs a SequentialMoE, maps to Mermaid config, and returns initialized Pipeline.
        """
        moe_module = SequentialMoE(
            experts=nn.ModuleList(experts) if isinstance(experts, list) else experts,
            thresholds=thresholds
        )
        
        mmd_lines = [
            "graph TD",
            "    %% Conceptual Cascade:",
        ]
        
        expert_count = len(experts) if type(experts) is list else len(experts)
        for i in range(expert_count):
            mmd_lines.append(f"    %% E{i}[Expert {i}]")
            if i < expert_count - 1:
                mmd_lines.append(f"    %% E{i} -- Low Confidence --> E{i+1}")
            else:
                mmd_lines.append(f"    %% E{i} --> OUT[Output]")
                
        mmd_lines.extend([
            "    %% execution DAG:",
            "    IN[Input] --> MOE_WRAPPER",
            "    MOE_WRAPPER --> OUT[Output]"
        ])
        
        mmd_LINES_string = "\n".join(mmd_lines)
        cls._save_mermaid(name, mmd_LINES_string)

        kwargs = pipeline_kwargs or {}
        return Pipeline(
            template_name=name,
            modules={"MOE_WRAPPER": moe_module},
            **kwargs
        )
