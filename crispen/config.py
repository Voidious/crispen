"""Load crispen configuration from pyproject.toml and optional .crispen.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CrispenConfig:
    """Runtime configuration for crispen."""

    # DuplicateExtractor: minimum statement weight for a sequence to be considered
    min_duplicate_weight: int = 3
    # DuplicateExtractor: maximum sequence length for duplicate search
    max_duplicate_seq_len: int = 8

    # TupleDataclass: minimum tuple element count to trigger replacement
    min_tuple_size: int = 4

    # LLM provider to use: "anthropic" (default), "moonshot", "openai", "deepseek",
    # "lmstudio", "ollama", "gemini", "zai", or "mistral"
    provider: str = "anthropic"
    # LLM model to use for all API calls
    model: str = "claude-sonnet-4-6"
    # Optional base URL override for OpenAI-compatible providers.
    # Useful for LM Studio when running on a non-default port, or for other
    # self-hosted OpenAI-compatible endpoints.
    base_url: Optional[str] = None
    # Optional tool_choice value for OpenAI-compatible providers.
    # When set, this string is sent as the tool_choice parameter instead of
    # the default named-function dict.  Use "required" for local models
    # (e.g. LM Studio / qwen3-8b) that do not support the named-function form.
    tool_choice: Optional[str] = None
    # HTTP timeout in seconds for each LLM API call.  A hard wall-clock limit
    # of api_timeout + 30 s is enforced on top of this to catch OS-level
    # blocking that the SDK timeout cannot interrupt.  Raise this when using a
    # slow local model (e.g. LM Studio on large models).
    api_timeout: float = 60.0
    # Whether to generate docstrings in extracted helper functions
    helper_docstrings: bool = False

    # FunctionSplitter: maximum function body lines (excluding docstring)
    max_function_length: int = 75

    # FileLimiter: maximum file line count before splitting is triggered.
    # Set to 0 to disable FileLimiter entirely.
    max_file_lines: int = 1000

    # Whether to update callers in diff files even if outside the diff ranges.
    # When False and unreachable callers exist, the transformation is skipped.
    update_diff_file_callers: bool = True

    # Number of additional extraction attempts after an algorithmic check fails.
    # 0 means no retry: the group is skipped on the first failure.
    extraction_retries: int = 2
    # Number of additional extraction attempts after the LLM verification step
    # rejects the output.  0 means no retry: the group is skipped on rejection.
    llm_verify_retries: int = 2

    # Number of additional FileLimiter attempts after an LLM-related failure
    # (e.g. LLM returned no placements, placement call failed, or code-gen
    # detected circular file imports).  0 means no retry.  Default 2 = three
    # total attempts.  Deterministic failures (single-SCC abort) are not retried.
    file_limiter_retries: int = 2

    # Number of additional attempts after an HTTP 429 (rate limit) response.
    # 0 means no retry: the error is raised immediately on the first 429.
    # Default 6 = seven total attempts.
    rate_limit_retries: int = 6
    # Initial wait time in seconds before the first retry after a 429 response.
    # Each subsequent retry doubles the delay (exponential backoff).
    # E.g. with the default 20s: 20, 40, 80, 160, 320, 640 seconds.
    rate_limit_backoff: float = 20.0

    # When True, any new file created by FileLimiter that still exceeds
    # max_file_lines is passed through FileLimiter again recursively.  The
    # recursion terminates naturally: each pass either reduces the file below
    # the limit, produces an abort (file cannot be split), or generates no
    # further new files over the limit.
    file_limiter_recursive: bool = True

    # When True and the diff covers every line of the file (whole-file add or
    # replacement), all new files created by FileLimiter are placed in a
    # subdirectory named after the source module (e.g. service/*.py for
    # service.py, or service/*.py for test_service.py after stripping the
    # leading test_ prefix).  For non-test files a package __init__.py is
    # generated in the subdirectory that re-exports the public API so callers
    # do not need to be updated.  For test files the original file keeps
    # re-export stubs; the new package directory is created alongside it.
    # Aborted if the target subdirectory already exists.
    file_limiter_subdir_split: bool = True

    # When True, pytest fixtures (functions decorated with @pytest.fixture or
    # @fixture) that are split out of a test file are routed to conftest.py in
    # the same directory instead of a regular sibling module.  pytest
    # auto-discovers fixtures from conftest.py, so no import is needed in the
    # original test file — eliminating the F401 "imported but unused" and F811
    # "redefinition of unused" flake8 warnings that arise when a fixture name
    # also appears as a test function parameter.  Set to False if you are not
    # using pytest or if your project manages fixtures in a non-standard way.
    file_limiter_pytest_conftest: bool = True

    # Controls when the original file keeps re-export stubs for public names
    # that were moved to new files.  Re-exports preserve the original module's
    # public API so existing callers need no changes.
    #
    # "always"      — Always add re-exports for every public name (most
    #                 conservative).  Best for library packages that publish a
    #                 public API not necessarily tested or imported from within
    #                 this codebase.
    # "application" — Add re-exports in non-test files (same as "always"), but
    #                 not in test files.  Pragmatic: application code keeps its
    #                 API intact; test modules are less likely to have external
    #                 callers.
    # "imported"    — Only add a re-export when the name is actually imported
    #                 from the original module somewhere else in the project
    #                 (the same rule already used for private names).  Best for
    #                 most application codebases.  May remove needed re-exports
    #                 for library packages whose public API is not exercised
    #                 within this codebase.  Note: only ``from module import
    #                 name`` style imports are detected; qualified access via
    #                 ``module.name`` is not scanned.
    file_limiter_reexports: str = "imported"

    # Controls how @patch string literals in test files are updated after
    # FileLimiter moves entities to new sub-modules.
    #
    # "ignore"  — Do nothing (default).  @patch strings are left as-is.
    # "basic"   — Scan every *.py file in the repo and replace string
    #             literals referencing a moved entity's old dotted path
    #             (e.g. "pkg.old_module.func") with the new path, but only
    #             when the update is unambiguous (non-forking).  For each
    #             entity, the patch target is the single new sub-module that
    #             imports it (the "caller"); if multiple new files import
    #             the entity (forking) it is skipped.  Import aliases from
    #             the original file that appear in exactly one new file are
    #             also updated.
    # "rewrite" — Performs "basic" updates first, then uses an LLM to
    #             resolve the remaining cases that basic mode skipped
    #             (entities that forked into multiple callers).  The LLM
    #             receives the original file, the diff, all new sub-files,
    #             and a chunk of test functions and proposes updated @patch
    #             strings.  A second LLM verify step checks each proposal;
    #             failed chunks are retried up to patch_update_retries
    #             additional times before being skipped.
    file_limiter_patch_update: str = "basic"

    # Number of additional LLM verify+retry attempts for each function chunk
    # in "rewrite" mode.  0 means no retry: a chunk rejected by the verify
    # LLM is skipped.  Default 2 = three total attempts (initial + two retries).
    patch_update_retries: int = 2

    # Maximum BFS hops from a test function to a target sub-module during
    # call-graph traversal (used by file_limiter_patch_update "basic" and
    # "rewrite" modes).  When this limit is reached the path is left unresolved
    # and a warning is printed to stderr.  Increase for very deep call chains.
    callgraph_max_depth: int = 12
    # Maximum distinct modules visited per resolution attempt during
    # call-graph traversal.  When this limit is reached the path is left
    # unresolved and a warning is printed to stderr.  Increase for very large
    # codebases with wide import graphs.
    callgraph_max_modules: int = 50

    # Refactor allow-list: if non-empty, only the named refactors are run.
    # Valid names: "if_not_else", "duplicate_extractor", "function_splitter",
    # "tuple_dataclass", "file_limiter", "match_function".
    # "match_function" controls the sub-pass inside duplicate_extractor that
    # replaces code blocks with calls to existing functions; it only takes
    # effect when duplicate_extractor is also running.
    # An empty list means "run all" (the default).
    enabled_refactors: List[str] = field(default_factory=list)
    # Refactor deny-list: named refactors are always skipped.
    # Ignored when enabled_refactors is non-empty.
    disabled_refactors: List[str] = field(default_factory=list)

    # Timing output level: "off" disables timing output entirely.
    # "basic" shows total run time, total LLM time, and total token counts.
    # "detailed" adds per-call-type, per-refactor, and per-file breakdowns.
    timing: str = "detailed"


def format_header(config: "CrispenConfig") -> List[str]:
    """Return config lines printed to stderr before the first LLM call."""
    w = 28
    lines = ["--- crispen ---"]
    lines.append(f"  {'provider:':<{w}}{config.provider}")
    lines.append(f"  {'model:':<{w}}{config.model}")
    lines.append(f"  {'api_timeout:':<{w}}{config.api_timeout}s")
    lines.append(f"  {'extraction_retries:':<{w}}{config.extraction_retries}")
    lines.append(f"  {'llm_verify_retries:':<{w}}{config.llm_verify_retries}")
    lines.append(f"  {'file_limiter_retries:':<{w}}{config.file_limiter_retries}")
    lines.append(f"  {'rate_limit_retries:':<{w}}{config.rate_limit_retries}")
    lines.append(f"  {'rate_limit_backoff:':<{w}}{config.rate_limit_backoff}s")
    lines.append(f"  {'file_limiter_reexports:':<{w}}{config.file_limiter_reexports}")
    lines.append(
        f"  {'file_limiter_patch_update:':<{w}}{config.file_limiter_patch_update}"
    )
    if config.file_limiter_patch_update == "rewrite":
        lines.append(f"  {'patch_update_retries:':<{w}}{config.patch_update_retries}")
    if config.base_url is not None:
        lines.append(f"  {'base_url:':<{w}}{config.base_url}")
    return lines


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
