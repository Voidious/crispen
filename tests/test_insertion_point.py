from crispen.refactors.duplicate_extractor import _find_insertion_point


def test_find_insertion_point_module_with_imports():
    source = "import os\nfrom sys import argv\n\ndef foo():\n    pass\n"
    # Should insert after the last import (index 1), so return 2
    assert _find_insertion_point(source, "<module>") == 2


def test_find_insertion_point_module_no_imports():
    source = "a = 1\n"
    # No imports: last_import stays -1, returns 0
    assert _find_insertion_point(source, "<module>") == 0
