"""
Standardized adapters for integrating external API models and Hugging Face models into the Sentinel System.
These adapters inherit from DLModule (or a compatible unified base) to plug seamlessly into the existing 
Pipeline and AI components.
"""

import json
import logging
from typing import Any, Dict, Optional, Union

# torch is an optional dependency for the ExternalModelAdapter
try:
    import torch
    TensorType = torch.Tensor
except ImportError:
    TensorType = Any

# Assuming DLModule is the standard base class for unified system integration
# Assuming DLModule is the standard base class for unified system integration
try:
    from ..utils.utils import DLModule
except ImportError:
    class DLModule:
        def __init__(self, *args, **kwargs): pass
        def forward(self, *args, **kwargs): pass

logger = logging.getLogger("SentinelExternalAdapters")


class ExternalModelAdapter(DLModule):
    """
    Base class for all external model adapters.
    Provides a unified interface for inference (forward pass) that system pipelines can rely on.
    """
    def __init__(self, name: str, api_key: Optional[str] = None):
        super().__init__()
        self.model_name = name
        self.api_key = api_key

    def forward(self, inputs: Union[str, Dict[str, Any], TensorType], **kwargs) -> Any:
        return self.predict(inputs, **kwargs)

    def predict(self, inputs: Union[str, Dict[str, Any], TensorType], **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement the predict method.")


class OpenAIAdapter(ExternalModelAdapter):
    """Adapter for OpenAI models (e.g., gpt-4, gpt-3.5-turbo)."""
    
    def __init__(self, name: str, api_key: str, organization: Optional[str] = None):
        super().__init__(name, api_key)
        self.organization = organization
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key, organization=self.organization)
        except ImportError:
            logger.warning("openai package not installed. OpenAIAdapter will fail on predict().")
            self.client = None

    def predict(self, inputs: Union[str, Dict[str, Any], TensorType], **kwargs) -> str:
        if not self.client:
            raise ImportError("Cannot run predict: 'openai' package is missing.")
            
        kwargs.setdefault("model", self.model_name)
        
        # Format string input to basic user message
        if isinstance(inputs, str):
            messages = [{"role": "user", "content": inputs}]
        elif isinstance(inputs, dict) and "messages" in inputs:
            messages = inputs["messages"]
        else:
            messages = [{"role": "user", "content": str(inputs)}]
            
        response = self.client.chat.completions.create(messages=messages, **kwargs)
        return response.choices[0].message.content


class AnthropicAdapter(ExternalModelAdapter):
    """Adapter for Anthropic models (e.g., claude-3-opus, claude-3-sonnet)."""
    
    def __init__(self, name: str, api_key: str):
        super().__init__(name, api_key)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            logger.warning("anthropic package not installed. AnthropicAdapter will fail on predict().")
            self.client = None

    def predict(self, inputs: Union[str, Dict[str, Any], TensorType], **kwargs) -> str:
        if not self.client:
            raise ImportError("Cannot run predict: 'anthropic' package is missing.")
            
        kwargs.setdefault("model", self.model_name)
        kwargs.setdefault("max_tokens", 1024)
        
        if isinstance(inputs, str):
            messages = [{"role": "user", "content": inputs}]
        elif isinstance(inputs, dict) and "messages" in inputs:
            messages = inputs["messages"]
        else:
            messages = [{"role": "user", "content": str(inputs)}]
            
        system = kwargs.pop("system", None)
        if system:
            response = self.client.messages.create(messages=messages, system=system, **kwargs)
        else:
            response = self.client.messages.create(messages=messages, **kwargs)
            
        return response.content[0].text


class GoogleAdapter(ExternalModelAdapter):
    """Adapter for Google Gemini/Vertex models."""
    
    def __init__(self, name: str, api_key: str):
        super().__init__(name, api_key)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            logger.warning("google-generativeai package not installed. GoogleAdapter will fail.")
            self.model = None

    def predict(self, inputs: Union[str, Dict[str, Any], TensorType], **kwargs) -> str:
        if not self.model:
            raise ImportError("Cannot run predict: 'google-generativeai' package is missing.")
            
        # Simplified: Google SDK expects a string or formatted content block
        query = inputs if isinstance(inputs, str) else str(inputs)
        response = self.model.generate_content(query, **kwargs)
        return response.text


class HuggingFaceAdapter(ExternalModelAdapter):
    """
    Adapter for Hugging Face models. 
    Can run entirely locally using transformers pipeline, or remotely via Inference API.
    """
    
    def __init__(self, name: str, api_key: Optional[str] = None, local: bool = False, task: str = "text-generation"):
        super().__init__(name, api_key)
        self.local = local
        self.task = task
        self.pipeline = None
        
        if self.local:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(self.task, model=self.model_name)
            except ImportError:
                logger.warning("transformers package missing. Local HuggingFaceAdapter will fail.")

    def predict(self, inputs: Union[str, Dict[str, Any], TensorType], **kwargs) -> str:
        if self.local:
            if not self.pipeline:
                raise ImportError("Transformers pipeline not initialized.")
            results = self.pipeline(inputs, **kwargs) # type: ignore
            # Extrapolate generated text from standard HF return formats
            if isinstance(results, list) and len(results) > 0 and "generated_text" in results[0]:
                return results[0]["generated_text"]
            return str(results)
        else:
            # Remote Inference API
            if not self.api_key:
                raise ValueError("API key required for remote Hugging Face inference.")
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            
            payload: Dict[str, Any] = {"inputs": inputs}
            payload.update(kwargs)
            
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            res_json = response.json()
            
            if isinstance(res_json, list) and len(res_json) > 0 and "generated_text" in res_json[0]:
                return res_json[0]["generated_text"]
            return str(res_json)


class GenericRESTAdapter(ExternalModelAdapter):
    """
    A highly flexible adapter allowing users to connect to ANY REST-based LLM API
    by specifying the URL, Headers, and JSON payload structure.
    """
    
    def __init__(
        self, 
        name: str, 
        endpoint: str, 
        headers: Dict[str, str], 
        payload_template: Dict[str, Any], 
        output_path: str = "choices[0].message.content"
    ):
        """
        :param endpoint: Full URL to the REST API endpoint.
        :param headers: Dictionary of HTTP headers (e.g., {"Authorization": "Bearer key"}).
        :param payload_template: A dictionary representing the required JSON body.
                                 Use the exact string "{INPUT}" as a placeholder for the user's prompt.
        :param output_path: Dot-notated string indicating how to extract the response text from the JSON reply.
                            e.g. "choices.0.message.content" or "response.text". Use brackets or dots.
        """
        super().__init__(name, api_key=None)
        self.endpoint = endpoint
        self.headers = headers
        self.payload_template = payload_template
        self.output_path = output_path
        
    def _inject_input(self, template: Any, input_text: str) -> Any:
        """Recursively injects the input_text into the template where '{INPUT}' is found."""
        if isinstance(template, str):
            return template.replace("{INPUT}", input_text)
        elif isinstance(template, list):
            return [self._inject_input(item, input_text) for item in template]
        elif isinstance(template, dict):
            return {k: self._inject_input(v, input_text) for k, v in template.items()}
        return template

    def _extract_output(self, response_json: Any, path: str) -> str:
        """Extracts the resulting string from the JSON using the output_path."""
        # Replace brackets with dots for uniform splitting
        clean_path = path.replace("[", ".").replace("]", "").replace("'", "").replace('"', "")
        keys = [k for k in clean_path.split(".") if k]
        
        curr = response_json
        try:
            for k in keys:
                if isinstance(curr, list):
                    curr = curr[int(k)]
                else:
                    curr = curr[k]
            return str(curr)
        except (KeyError, IndexError, TypeError):
            logger.error(f"Failed to extract '{path}' from {response_json}")
            return str(response_json)

    def predict(self, inputs: Union[str, Dict[str, Any], TensorType], **kwargs) -> str:
        import requests
        
        # Process input ensuring it's a string
        input_str = inputs if isinstance(inputs, str) else json.dumps(inputs)
        
        # Inject the input into the defined payload template
        payload = self._inject_input(self.payload_template, input_str)
        
        # Merge any kwargs directly into payload if users want to override parameters dynamically.
        if isinstance(payload, dict):
            payload.update(kwargs)
            
        try:
            res = requests.post(self.endpoint, headers=self.headers, json=payload)
            res.raise_for_status()
            res_json = res.json()
            return self._extract_output(res_json, self.output_path)
        except Exception as e:
            logger.error(f"GenericRESTAdapter execution failed: {e}")
            raise
