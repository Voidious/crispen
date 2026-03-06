from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.entity_parser import Entity, EntityKind
from crispen.file_limiter.runner import _verify_preservation
from .helpers import _make_entity


def test_verify_entity_source_in_original():
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="def foo():\n    pass\n",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source, []) == []


def test_verify_entity_source_in_new_file():
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={"utils.py": "def foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source, []) == []


def test_verify_entity_source_missing():
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="# nothing relevant\n",
        abort=False,
    )
    failures = _verify_preservation([entity], split, post_source, [])
    assert len(failures) == 1
    assert "'foo'" in failures[0]
    assert "1" in failures[0]  # start line
    assert "2" in failures[0]  # end line


def test_verify_entity_source_missing_long():
    # Entity with more than 3 lines → preview includes trailing "..."
    post_source = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    pass\n"
    entity = _make_entity("foo", 1, 5)
    split = SplitResult(
        new_files={},
        original_source="# nothing relevant\n",
        abort=False,
    )
    failures = _verify_preservation([entity], split, post_source, [])
    assert len(failures) == 1
    assert "..." in failures[0]


def test_verify_empty_entity_source_skipped():
    # Entity spanning only a blank line → rstrip → "" → falsy → skipped.
    post_source = "\n"
    entity = _make_entity("_block_1", 1, 1)
    split = SplitResult(
        new_files={},
        original_source="# completely different",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source, []) == []


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
    assert _verify_preservation([entity], split, post_source, []) == []


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
    failures = _verify_preservation([entity], split, post_source, placements)
    assert len(failures) == 1
    assert "migrated" in failures[0]
    assert "utils.py" in failures[0]


def test_verify_annotation_stayed():
    # Failure for an entity not in any placement → annotated "stayed in original".
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="# empty",
        abort=False,
    )
    failures = _verify_preservation([entity], split, post_source, [])
    assert len(failures) == 1
    assert "stayed in original" in failures[0]


def test_verify_pruned_inline_import_passes():
    # Entity has an inline import; the new file has it pruned to a top-level one.
    # Both sides are stripped before comparison, so the match succeeds.
    post_source = "def foo():\n    import os\n    return os.getcwd()\n"
    entity = _make_entity("foo", 1, 3)
    split = SplitResult(
        new_files={"utils.py": "import os\n\ndef foo():\n    return os.getcwd()"},
        original_source="# original\n",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source, []) == []


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
    assert _verify_preservation([entity], split, post_source, []) == []


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
    assert _verify_preservation([entity], split, post_source, []) == []


def test_verify_async_def_entity_passes():
    # Async functions are found after import stripping (no imports involved).
    post_source = "async def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={"utils.py": "async def foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source, []) == []
