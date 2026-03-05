from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

# Matches any line that is an import statement (plain or from-import).
_IMPORT_LINE_RE = re.compile(r"^(import\s+|from\s+\S.*\s+import\s+)")

# Matches a `from __future__ import …` line (with optional trailing newline).
_FUTURE_IMPORT_LINE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)


def _find_project_root(path: Path) -> Optional[Path]:
    """Walk up from *path* to find the project root directory.

    Returns the first directory containing ``pyproject.toml``, ``setup.py``,
    ``setup.cfg``, or ``.git``.  Returns ``None`` when the filesystem root is
    reached without finding any of these markers.
    """
    markers = {"pyproject.toml", "setup.py", "setup.cfg", ".git"}
    current = path if path.is_dir() else path.parent
    while True:
        if any((current / m).exists() for m in markers):
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _module_path_from_file(project_root: Path, file_path: Path) -> Optional[str]:
    """Return the dotted Python module path of *file_path* relative to *project_root*.

    Returns ``None`` when *file_path* is not under *project_root*.
    """
    try:
        rel = file_path.relative_to(project_root)
    except ValueError:
        return None
    return ".".join(rel.with_suffix("").parts)


def _abs_package_for_dir(file_path: str) -> Optional[str]:
    """Return the dotted package path of the directory containing *file_path*.

    Used to generate absolute imports for test files so that pytest's default
    import mode (which loads test files as top-level modules, not package
    members) does not choke on ``from .module import …`` syntax.

    Returns an empty string for files sitting directly in the project root,
    ``None`` when the project root cannot be determined.
    """
    orig = Path(file_path).resolve()
    project_root = _find_project_root(orig.parent)
    if project_root is None:
        return None
    try:
        rel = orig.parent.relative_to(project_root)
    except ValueError:
        return None
    return ".".join(rel.parts)
