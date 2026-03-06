from __future__ import annotations
from .split_core import _REL_IMPORT_RE


def _bump_relative_imports(source: str) -> str:
    """Increment the level of every relative import in *source* by one.

    Used when file content is moved one directory level deeper, e.g. when the
    source originally written for ``pkg/module.py`` becomes the content of
    ``pkg/module/__init__.py``, or when new files go into a subdirectory
    package instead of sitting next to the original file.

    ``from .foo`` → ``from ..foo``, ``from ..bar`` → ``from ...bar``, etc.
    Absolute imports are not affected.
    """
    return _REL_IMPORT_RE.sub(lambda m: f"from .{m.group(1)}", source)
