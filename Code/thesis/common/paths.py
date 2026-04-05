from pathlib import Path


def project_root() -> Path:
    """Repository root (parent of Code/)."""
    return Path(__file__).resolve().parents[3]


def thesis_dir() -> Path:
    return Path(__file__).resolve().parents[1]
