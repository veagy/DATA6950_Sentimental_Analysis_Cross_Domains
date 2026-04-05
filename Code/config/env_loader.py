import os
from pathlib import Path

# Project root (assuming config lives at src/config/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Mapping: .env variable name -> standard env var name(s) used by libraries
_ENV_KEY_MAPPING = {
    "WANDB_KEY": ["WANDB_API_KEY"],
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "HF_TOKEN": ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
}


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse .env file without external deps."""
    result = {}
    if not env_path.exists():
        return result
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def load_api_keys(env_path: Path | None = None, verbose: bool = False) -> bool:
    if env_path is None:
        env_path = _PROJECT_ROOT / ".env"

    if not env_path.exists():
        return False

    try:
        from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

        load_dotenv(env_path, override=False)
        env_vars = dict(os.environ)
    except ImportError:
        env_vars = _parse_env_file(env_path)

    loaded = False
    for env_key, target_vars in _ENV_KEY_MAPPING.items():
        value = env_vars.get(env_key) or os.environ.get(env_key)
        if value:
            for target in target_vars:
                os.environ[target] = value
            loaded = True
            if verbose:
                print(f"[env_loader] Loaded {env_key} -> {target_vars[0]}")

    return loaded


def get_api_key(name: str) -> str | None:
    lookup = {
        "wandb": "WANDB_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "hf": "HF_TOKEN",
    }
    var = lookup.get(name.lower())
    return os.environ.get(var) if var else None
