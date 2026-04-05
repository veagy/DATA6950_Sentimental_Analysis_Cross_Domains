"""
Storage-agnostic data source loader for the Sentinel Data Package.
Uses the centralized data_base.py engine for 30+ format support and cloud integration.
"""

from ..train.utils.data_source import load_dataframe, make_loader_from_source

# Exposed for Data Package consumers
__all__ = ["load_dataframe", "make_loader_from_source"]
