from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.runner import _strip_imports_by_line, _verify_preservation
from .helpers import _make_entity


def test_verify_pruned_inline_import_passes():
    # Entity has an inline import; the new file has it pruned to a top-level one.
    # Both sides are stripped before comparison, so the match succeeds.
    # verified_lines counts only the non-import lines of the migrated entity.
    post_source = "def foo():\n    import os\n    return os.getcwd()\n"
    entity = _make_entity("foo", 1, 3)
    split = SplitResult(
        new_files={"utils.py": "import os\n\ndef foo():\n    return os.getcwd()"},
        original_source="# original\n",
        abort=False,
    )
    placements = [GroupPlacement(group=["foo"], target_file="utils.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
    # "def foo():\n    return os.getcwd()" → 2 lines (import stripped)
    assert vr.verified_lines == 2


def test_verify_inline_import_not_pruned_also_passes():
    # Import was NOT pruned — it appears on both sides. Stripping both sides
    # still produces a match.
    post_source = "def foo():\n    import os\n    return os.getcwd()\n"
    entity = _make_entity("foo", 1, 3)
    split = SplitResult(
        new_files={"utils.py": "def foo():\n    import os\n    return os.getcwd()"},
        original_source="# original\n",
        abort=False,
    )
    placements = [GroupPlacement(group=["foo"], target_file="utils.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
    assert vr.verified_lines == 2


def test_verify_multiline_import_stripped():
    # Multi-line imports are removed correctly using AST line spans.
    post_source = (
        "def foo():\n"
        "    from os import (\n"
        "        path,\n"
        "        getcwd,\n"
        "    )\n"
        "    return getcwd()\n"
    )
    entity = _make_entity("foo", 1, 6)
    # New file has the multi-line import removed (3 lines gone).
    split = SplitResult(
        new_files={
            "utils.py": "from os import path, getcwd\n\ndef foo():\n    return getcwd()"
        },
        original_source="# original\n",
        abort=False,
    )
    placements = [GroupPlacement(group=["foo"], target_file="utils.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
    # "def foo():\n    return getcwd()" → 2 lines (4-line import stripped)
    assert vr.verified_lines == 2


def test_strip_imports_no_imports():
    src = "def foo():\n    return 1\n"
    assert _strip_imports_by_line(src) == src


def test_strip_imports_single_line():
    src = "import os\nx = 1\n"
    assert _strip_imports_by_line(src) == "x = 1\n"


def test_strip_imports_multiline():
    src = "from os import (\n    path,\n    getcwd,\n)\nx = 1\n"
    assert _strip_imports_by_line(src) == "x = 1\n"


def test_strip_imports_inner_import():
    # Imports inside a function body are also stripped.
    src = "def foo():\n    import os\n    return os.getcwd()\n"
    assert _strip_imports_by_line(src) == "def foo():\n    return os.getcwd()\n"


def test_strip_imports_syntax_error_returns_unchanged():
    src = "def foo(:\n    pass\n"
    assert _strip_imports_by_line(src) == src
