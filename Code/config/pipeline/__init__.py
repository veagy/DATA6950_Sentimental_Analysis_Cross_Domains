# Pipeline configuration package.
# Mermaid flowcharts stored in ./mermaid/

import json
from pathlib import Path

_PATH_PIPELINE = Path(__file__).resolve().parent
SENTINEL_CONFIG_PATH = _PATH_PIPELINE / "sentinel_config.json"
PATH_MERMAID = _PATH_PIPELINE / "mermaid"


def load_sentinel_config() -> dict:
    """Load pipeline sentinel_config.json if it exists."""
    if SENTINEL_CONFIG_PATH.exists():
        try:
            return json.loads(SENTINEL_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}
