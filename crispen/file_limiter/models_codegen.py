from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`


@dataclass
class SplitResult:
    """Output of :func:`generate_file_splits`."""

    new_files: Dict[str, str]  # {target_file: source_code}
    original_source: str  # updated original file source
    abort: bool  # True if generation failed / nothing to split
    abort_reason: str = ""  # human-readable explanation when abort=True
