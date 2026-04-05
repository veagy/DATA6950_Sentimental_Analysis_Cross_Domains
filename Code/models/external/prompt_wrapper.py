"""
Prompt Workflow Layer for Sentinel System.
Intercepts raw user inputs, loads standardized Prompt Templates, injects context, 
and forwards the structured prompt to the configured underlying model (local or external).
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .adapters import ExternalModelAdapter
from ..utils.utils import DLModule

logger = logging.getLogger("SentinelPromptWrapper")

PROMPTS_DIR = Path("./prompts")


class PromptWorkflowWrapper(DLModule):
    """
    Acts as an intermediary between the user input and the ML/LLM models.
    Loads templates dynamically, injects variables, and normalizes output.
    """
    
    def __init__(self, backend_model: Union[DLModule, ExternalModelAdapter], template_name: str, **default_vars):
        """
        :param backend_model: The instantiated model class (e.g., OpenAIAdapter, or local DLModule)
        :param template_name: The filename (without extension) in .configs/prompts to load
        :param default_vars: Default variables to inject into the template
        """
        super().__init__()
        self.backend_model = backend_model
        self.template_name = template_name
        self.default_vars = default_vars
        self.template_content = self._load_template(template_name)
        
    def _load_template(self, template_name: str) -> str:
        """Loads a text template from the .configs/prompts directory."""
        if not PROMPTS_DIR.exists():
            PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
            
        # Support multiple extensions
        for ext in [".txt", ".md", ".json"]:
            filepath = PROMPTS_DIR / f"{template_name}{ext}"
            if filepath.exists():
                try:
                    return filepath.read_text(encoding="utf-8")
                except OSError as e:
                    logger.error(f"Failed to read prompt template {filepath}: {e}")
                    
        # Fallback if no template is found
        logger.warning(f"Template '{template_name}' not found in {PROMPTS_DIR}. Using passthrough fallback.")
        return "{USER_INPUT}"

    def _format_prompt(self, user_input: str, dynamic_vars: Dict[str, Any]) -> str:
        """Injects variables into the prompt template."""
        combined_vars = {**self.default_vars, **dynamic_vars}
        
        # Always ensure USER_INPUT is available
        combined_vars["USER_INPUT"] = user_input
        
        formatted = self.template_content
        for key, value in combined_vars.items():
            placeholder = "{" + key + "}"
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(value))
                
        # If the template was literally just passthrough because it wasn't found
        if formatted == "{USER_INPUT}":
            return user_input
            
        return formatted

    def forward(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> Any:
        return self.predict(inputs, **kwargs)

    def predict(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """
        Takes the user input, wraps it in the template, and sends it to the backend model.
        """
        dynamic_vars = kwargs.pop("prompt_vars", {})
        
        raw_input = ""
        if isinstance(inputs, dict):
            # If the input itself is a dictionary, use it to update variables
            dynamic_vars.update(inputs)
            raw_input = inputs.get("query", inputs.get("input", str(inputs)))
        else:
            raw_input = str(inputs)
            
        # Create the structured prompt
        final_prompt = self._format_prompt(raw_input, dynamic_vars)
        
        # Log the final prompt in debug mode
        logger.debug(f"Structured Prompt being sent to {self.backend_model.__class__.__name__}:\n{final_prompt}")
        
        # Execute the model mapping
        try:
            if hasattr(self.backend_model, "predict"):
                return self.backend_model.predict(final_prompt, **kwargs)
            elif hasattr(self.backend_model, "forward"):
                return self.backend_model.forward(final_prompt, **kwargs)
            else:
                return self.backend_model(final_prompt, **kwargs)
        except Exception as e:
            logger.error(f"PromptWrapper failed executing backend model: {e}")
            raise
