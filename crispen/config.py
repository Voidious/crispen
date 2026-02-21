"""Load crispen configuration from pyproject.toml and optional .crispen.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CrispenConfig:
    """Runtime configuration for crispen."""

    # DuplicateExtractor: minimum statement weight for a sequence to be considered
    min_duplicate_weight: int = 3
    # DuplicateExtractor: maximum sequence length for duplicate search
    max_duplicate_seq_len: int = 8

    # TupleDataclass: minimum tuple element count to trigger replacement
    min_tuple_size: int = 4

    # LLM provider to use: "anthropic" (default) or "moonshot"
    provider: str = "anthropic"
    # LLM model to use for all API calls
    model: str = "claude-sonnet-4-6"
    # Whether to generate docstrings in extracted helper functions
    helper_docstrings: bool = False

    # Whether to update callers in diff files even if outside the diff ranges.
    # When False and unreachable callers exist, the transformation is skipped.
    update_diff_file_callers: bool = True


def _read_toml(path: Path) -> dict:
    """Read a TOML file; return empty dict if missing or unparseable."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _apply(cfg: CrispenConfig, d: dict) -> None:
    """Overlay dict values onto cfg, ignoring unknown keys."""
    valid = set(cfg.__dataclass_fields__)
    for key, val in d.items():
        if key in valid:
            setattr(cfg, key, val)


def load_config(project_root: Optional[Path] = None) -> CrispenConfig:
    """Load config from pyproject.toml [tool.crispen], then .crispen.toml."""
    if project_root is None:
        project_root = Path.cwd()
    cfg = CrispenConfig()
    pyproject = _read_toml(project_root / "pyproject.toml")
    _apply(cfg, pyproject.get("tool", {}).get("crispen", {}))
    local = _read_toml(project_root / ".crispen.toml")
    _apply(cfg, local)
    return cfg
