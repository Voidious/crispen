from crispen.refactors.duplicate_extractor import _build_helper_insertion


def test_build_helper_insertion_absorbs_blank_before_function():
    # Blank line between import and def is absorbed; 2 blank lines ensured.
    source = "import os\n\ndef foo():\n    pass\n"
    lines = source.splitlines(keepends=True)
    helper = "def _helper():\n    pass\n"
    start, end, text = _build_helper_insertion(lines, 2, helper, "module_level")
    # Blank line at index 1 is before insert_pos=2, so before_blanks=1 → start=1.
    assert start == 1
    assert end == 2  # no blanks after insert_pos (def foo starts there)
    assert text.startswith("\n\n")
    assert text.endswith("\n\n")
    assert "def _helper():" in text


def test_build_helper_insertion_no_surrounding_blanks():
    # No blanks to absorb → pure insertion with 2 blank lines each side.
    source = "import os\ndef foo():\n    pass\n"
    lines = source.splitlines(keepends=True)
    helper = "def _helper():\n    pass\n"
    start, end, text = _build_helper_insertion(lines, 1, helper, "module_level")
    assert start == 1
    assert end == 1  # pure insertion
    assert text.startswith("\n\n")
    assert text.endswith("\n\n")


def test_build_helper_insertion_staticmethod_uses_one_blank():
    # Staticmethod placement: 1 blank line before and after.
    source = "class Foo:\n    def bar(self):\n        pass\n"
    lines = source.splitlines(keepends=True)
    helper = "    @staticmethod\n    def _h():\n        pass\n"
    start, end, text = _build_helper_insertion(lines, 1, helper, "staticmethod:Foo")
    assert start == 1
    assert end == 1  # no blanks to absorb
    assert text.startswith("\n")
    assert not text.startswith("\n\n")
    assert text.endswith("\n\n")  # clean + 1 trailing blank = \n + \n
