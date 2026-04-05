from typing import Optional, Dict, Any, Union
from pathlib import Path
from ....models.models import Pipeline
from .hrm_model import HierarchicalReasoningModel, HRMClassifierWrapper, HRMConfig
PATH_USER_CONFIGS_MODELS = Path("./configs/models")

class HRMPipelineBuilder:
    """
    Builder class for creating Hierarchical Reasoning Model pipelines.
    Generates the underlying PyTorch HRM module and its corresponding 
    system Mermaid configuration diagram for storage and execution.
    """

    @staticmethod
    def _save_template(template_name: str, mmd_content: str) -> None:
        """Helper to save the generated Mermaid to the user configurations directory."""
        if not PATH_USER_CONFIGS_MODELS.exists():
            PATH_USER_CONFIGS_MODELS.mkdir(parents=True, exist_ok=True)
        out_path = PATH_USER_CONFIGS_MODELS / f"{template_name}.mmd"
        out_path.write_text(mmd_content, encoding="utf-8")

    @classmethod
    def create_hrm_pipeline(
        cls,
        template_name: str,
        hrm_config: Union[Dict, HRMConfig],
        n_classes: int = 2
    ) -> Pipeline:
        """
        Create a flexible Hierarchical Reasoning Model pipeline.

        Parameters
        ----------
        template_name : str
            The name under which the `.mmd` configuration will be saved.
        hrm_config : Union[Dict, HRMConfig]
            The configuration parameters spanning `H_cycles`, `L_cycles`, 
            sub-model type strings (e.g., 'DecoderLM', 'ViT'), and `hidden_size`.
        n_classes: int
            Number of output classes for the HRM classification head.
        pipeline_kwargs : dict
            Additional kwargs to pass to the final `Pipeline` instantiation 
            (e.g., `device`, `dtype`).

        Returns
        -------
        Pipeline
            An interactive system `Pipeline` wrapping the constructed HRM module.
        """
        # 1. Instantiate Core
        enc = HierarchicalReasoningModel(config=hrm_config)
        hrm_module = HRMClassifierWrapper(enc, n_classes)

        # 2. Build Structural Mermaid execution DAG
        # Similar to the MoE pattern, we simplify the Pipeline-executed DAG
        # exposing the complex iterative sequence within the PyTorch forward pass.
        
        # We explicitly represent the nested looping purely for conceptual logging
        config_dict = hrm_config if isinstance(hrm_config, dict) else hrm_config.__dict__
        
        # Simplified execution graph 
        mmd_content = f"""%% Hierarchical Reasoning Model (HRM)
%% This wrapper seamlessly loops execution {config_dict.get('H_cycles', 2)} H-Cycles 
%% and {config_dict.get('L_cycles', 3)} L-Cycles across '{config_dict.get('h_level_model', 'DecoderLM')}' 
%% and '{config_dict.get('l_level_model', 'DecoderLM')}' base sub-models.

graph TD
    IN[Input] --> HRM_CORE
    HRM_CORE --> OUT[Output]
"""
        cls._save_template(template_name, mmd_content)

        # 3. Inject into generalized system Pipeline wrapper
        # The `modules={"HRM_CORE": hrm_module}` dictionary bridges the conceptual 
        # Mermaid node to the physical PyTorch Module instantiated above.
        return Pipeline(
            mermaid_flowchart=mmd_content,
            modules={"HRM_CORE": hrm_module}
        )
