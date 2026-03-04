from __future__ import annotations
from crispen.file_limiter.code_gen import _collect_external_imported_names


def test_collect_external_imported_names_relative_path():
    # Non-absolute path → empty set (no scan).
    assert _collect_external_imported_names("relative/path.py") == set()
