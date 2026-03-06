from __future__ import annotations
from crispen.file_limiter.code_gen import _relative_import_prefix


def test_relative_import_prefix_same_directory():
    # Both files at the root level → single dot.
    assert _relative_import_prefix("a.py", "b.py") == ".b"


def test_relative_import_prefix_sibling_subdir():
    # from_file is in sub/, to_file is in helpers/ → go up one, then down.
    assert _relative_import_prefix("sub/a.py", "helpers/b.py") == "..helpers.b"


def test_relative_import_prefix_same_subdir():
    # Both in the same subdirectory → single dot.
    assert _relative_import_prefix("sub/a.py", "sub/b.py") == ".b"


def test_relative_import_prefix_to_nested():
    # to_file is in a subdirectory of root while from_file is at root.
    assert _relative_import_prefix("a.py", "helpers/b.py") == ".helpers.b"
