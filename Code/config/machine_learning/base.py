"""
Re-exports from Code.config for backward compatibility.

Use: from Code.config.machine_learning.base import ConfigTemplate
Or: from Code.config import ConfigTemplate
"""

from ...config.base import ConfigTemplate, JsonConfig, create_model, resolve_config_value

__all__ = ["ConfigTemplate", "JsonConfig", "create_model", "resolve_config_value"]
