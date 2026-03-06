from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.entity_parser import Entity, EntityKind
from crispen.file_limiter.runner import _strip_imports_by_line, _verify_preservation
from .test_planning_and_execution import _make_entity


def test_verify_entity_source_in_original():
    # Entity that stayed in the original file — passes verification but is not
    # counted (it wasn't a FileLimiter edit).
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="def foo():\n    pass\n",
        abort=False,
    )
    vr = _verify_preservation([entity], split, post_source, [])
    assert vr.failures == []
    assert vr.verified_functions == 0
    assert vr.verified_lines == 0


def test_verify_entity_source_in_new_file():
    # Entity that was migrated — passes verification and is counted.
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={"utils.py": "def foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )
    placements = [GroupPlacement(group=["foo"], target_file="utils.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
    assert vr.verified_functions == 1
    assert vr.verified_lines == 2  # "def foo():\n    pass" → 2 lines matched


def test_verify_entity_source_missing():
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="# nothing relevant\n",
        abort=False,
    )
    vr = _verify_preservation([entity], split, post_source, [])
    assert len(vr.failures) == 1
    assert "'foo'" in vr.failures[0]
    assert "1" in vr.failures[0]  # start line
    assert "2" in vr.failures[0]  # end line
    assert vr.verified_lines == 0


def test_verify_entity_source_missing_long():
    # Entity with more than 3 lines → preview includes trailing "..."
    post_source = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    pass\n"
    entity = _make_entity("foo", 1, 5)
    split = SplitResult(
        new_files={},
        original_source="# nothing relevant\n",
        abort=False,
    )
    vr = _verify_preservation([entity], split, post_source, [])
    assert len(vr.failures) == 1
    assert "..." in vr.failures[0]


def test_verify_empty_entity_source_skipped():
    # Entity spanning only a blank line → rstrip → "" → falsy → skipped.
    post_source = "\n"
    entity = _make_entity("_block_1", 1, 1)
    split = SplitResult(
        new_files={},
        original_source="# completely different",
        abort=False,
    )
    vr = _verify_preservation([entity], split, post_source, [])
    assert vr.failures == []
    assert vr.verified_lines == 0


def test_verify_top_level_entity_skipped():
    # TOP_LEVEL entities (import/docstring blocks) are always skipped —
    # they are intentionally restructured when the file is split.
    post_source = "from __future__ import annotations\nimport os\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["annotations", "os"])
    split = SplitResult(
        new_files={},
        original_source="# completely different",
        abort=False,
    )
    vr = _verify_preservation([entity], split, post_source, [])
    assert vr.failures == []
    assert vr.verified_lines == 0


def test_verify_annotation_migrated():
    # Failure for an entity that was in the plan → annotated "migrated → target".
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={"utils.py": "# empty"},
        original_source="# empty",
        abort=False,
    )
    placements = [GroupPlacement(group=["foo"], target_file="utils.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert len(vr.failures) == 1
    assert "migrated" in vr.failures[0]
    assert "utils.py" in vr.failures[0]


def test_verify_annotation_stayed():
    # Failure for an entity not in any placement → annotated "stayed in original".
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="# empty",
        abort=False,
    )
    vr = _verify_preservation([entity], split, post_source, [])
    assert len(vr.failures) == 1
    assert "stayed in original" in vr.failures[0]


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


def test_verify_async_def_entity_passes():
    # Async functions are found after import stripping (no imports involved).
    post_source = "async def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={"utils.py": "async def foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )
    placements = [GroupPlacement(group=["foo"], target_file="utils.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
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
