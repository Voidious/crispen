from __future__ import annotations
from pathlib import Path


def _relative_import_prefix(from_file: str, to_file: str) -> str:
    """Return the Python relative-import prefix for *to_file* as seen from *from_file*.

    Both paths are relative to the same base directory (the original file's
    directory).  Examples::

        _relative_import_prefix("utils.py", "helpers.py")          → ".helpers"
        _relative_import_prefix("sub/a.py", "helpers/b.py")        → "..helpers.b"
        _relative_import_prefix("sub/a.py", "sub/b.py")            → ".b"
    """
    from_parts = Path(from_file).parent.parts  # () for top-level files
    to_module_parts = Path(to_file).with_suffix("").parts  # ("helpers", "b")
    to_dir_parts = Path(to_file).parent.parts  # ("helpers",)

    # Length of the common directory prefix between from_dir and to_dir.
    common_len = 0
    for fp, tp in zip(from_parts, to_dir_parts):
        if fp == tp:
            common_len += 1
        else:
            break

    levels_up = len(from_parts) - common_len
    module = ".".join(to_module_parts[common_len:])
    return "." * (levels_up + 1) + module


def _target_module_name(target_file: str) -> str:
    """Convert a relative target filename to a dotted module name.

    ``"utils.py"`` → ``"utils"``, ``"helpers/io.py"`` → ``"helpers.io"``.
    """
    path = Path(target_file)
    parts = list(path.with_suffix("").parts)
    return ".".join(parts)
