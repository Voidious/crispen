from crispen.refactors.duplicate_extractor import _find_insertion_point


def test_find_insertion_point_indented_func_at_file_start():
    # Edge case: the target def has method_indent > 0 but is at line 0 so the
    # backward-search loop range is empty.  Falls through to decorator walk
    # which also exits immediately (j=-1), returning 0.
    source = "    def inner():\n        pass\n"
    # "def inner" found at i=0 (indent=4).  range(-1, -1, -1) is empty → loop
    # body never runs → fall through to decorator walk → j = -1 → return 0.
    assert _find_insertion_point(source, "inner") == 0
