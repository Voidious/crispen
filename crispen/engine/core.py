from typing import Dict, List, NamedTuple, Optional, Set, Tuple
from libcst.metadata import MetadataWrapper
import libcst as cst
from ..stats import RunStats
from ..config import CrispenConfig
from ..errors import CrispenAPIError
from ..refactors.duplicate_extractor import DuplicateExtractor
from ..refactors.function_splitter import FunctionSplitter
from ..refactors.if_not_else import IfNotElse
from ..refactors.tuple_dataclass import TupleDataclass


# Single-file refactors applied in order before TupleDataclass.
_REFACTORS = [IfNotElse, DuplicateExtractor, FunctionSplitter]

# Refactor keys that invoke LLM calls (used to decide whether to print config).
_LLM_REFACTOR_KEYS = frozenset(
    {"duplicate_extractor", "function_splitter", "tuple_dataclass", "file_limiter"}
)

# Canonical snake_case name for each refactor class (used by _should_run).
_REFACTOR_KEY: Dict[type, str] = {
    IfNotElse: "if_not_else",
    DuplicateExtractor: "duplicate_extractor",
    FunctionSplitter: "function_splitter",
}


def _should_run(name: str, config: CrispenConfig) -> bool:
    """Return True if the named refactor should run given the config.

    When ``config.enabled_refactors`` is non-empty only names in that list run.
    Otherwise names in ``config.disabled_refactors`` are skipped.
    """
    if config.enabled_refactors:
        return name in config.enabled_refactors
    return name not in config.disabled_refactors


# Directory names excluded from the outside-caller scan (e.g. virtual environments).
_EXCLUDED_DIR_NAMES = frozenset(
    {".venv", "venv", "env", ".tox", "__pycache__", "node_modules"}
)

# Total wall-clock budget for all files in _find_outside_callers (seconds).
_SCOPE_ANALYSIS_TIMEOUT = 10


class _ApplyResult(NamedTuple):
    """Return type of _apply_tuple_dataclass."""

    source: str
    msgs: List[str]
    td: Optional[TupleDataclass]


def _apply_tuple_dataclass(
    filepath: str,
    ranges: List[Tuple[int, int]],
    source: str,
    verbose: bool,
    approved_public_funcs: Set[str],
    min_size: int = 4,
    blocked_scopes: Optional[Set[str]] = None,
) -> "_ApplyResult":
    """Run TupleDataclass on *source*. Returns (new_source, messages, transformer)."""
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        return _ApplyResult(
            source, [f"SKIP {filepath} (TupleDataclass): parse error: {exc}"], None
        )

    wrapper = MetadataWrapper(tree)
    try:
        td = TupleDataclass(
            ranges,
            source=source,
            verbose=verbose,
            approved_public_funcs=approved_public_funcs,
            min_size=min_size,
            blocked_scopes=blocked_scopes,
        )
        new_tree = wrapper.visit(td)
    except CrispenAPIError:
        raise
    except Exception as exc:
        return _ApplyResult(
            source,
            [f"SKIP {filepath} (TupleDataclass): transform error: {exc}"],
            None,
        )

    new_source = td.get_rewritten_source() or new_tree.code
    if new_source == source:
        return _ApplyResult(source, [], td)

    try:
        compile(new_source, filepath, "exec")
    except SyntaxError as exc:  # pragma: no cover
        return _ApplyResult(
            source,
            [f"SKIP {filepath} (TupleDataclass): output not valid Python: {exc}"],
            td,
        )

    msgs = [f"{filepath}: {m}" for m in td.get_changes()]
    return _ApplyResult(new_source, msgs, td)


def _categorize_into_stats(stats: RunStats, msg: str) -> None:
    """Increment the appropriate counter in *stats* for a raw change message."""
    if msg.startswith("IfNotElse:"):
        stats.if_not_else += 1
    elif msg.startswith("TupleDataclass:"):
        stats.tuple_to_dataclass += 1
    elif msg.startswith("DuplicateExtractor:") and "with call to" in msg:
        stats.duplicate_matched += 1
    elif msg.startswith("DuplicateExtractor:"):
        stats.duplicate_extracted += 1
    elif msg.startswith("split "):
        stats.function_split += 1
