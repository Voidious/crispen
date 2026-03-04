from crispen.refactors.duplicate_extractor import _find_insertion_point


def test_find_insertion_point_function_found():
    source = "import os\n\ndef target():\n    pass\n"
    # def target is at line index 2
    assert _find_insertion_point(source, "target") == 2


def test_find_insertion_point_function_not_found():
    source = "a = 1\n"
    # Falls back to 0
    assert _find_insertion_point(source, "missing_func") == 0
