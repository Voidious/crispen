from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.entity_parser import Entity, EntityKind
from crispen.file_limiter.runner import _strip_imports_by_line, _verify_preservation
from .test_runner_core import _make_entity


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


def test_verify_blank_line_collapse_after_pruning_passes():
    # Regression: when multiple consecutive inline imports are all pruned to
    # top-level, the resulting consecutive blank lines (3+ newlines before
    # indented content) are collapsed by _normalize_blank_lines in code_gen.
    # Verification must apply the same normalization to entity_no_imports so
    # the substring match doesn't fail due to a blank-line count mismatch.
    post_source = (
        "def test_seq():\n"
        '    """Docstring."""\n'
        "    import libcst as cst\n"
        "\n"
        "    from libcst.metadata import MetadataWrapper\n"
        "\n"
        "    from foo import Bar\n"
        "\n"
        "    x = cst.parse_module('')\n"
        "    w = MetadataWrapper(x)\n"
        "    b = Bar()\n"
    )
    entity = _make_entity("test_seq", 1, 13)
    # New file has all 3 inline imports hoisted to top-level and pruned from
    # the function body; _normalize_blank_lines collapsed the 3+ consecutive
    # blank lines down to 1.
    new_file_src = (
        "import libcst as cst\n"
        "from libcst.metadata import MetadataWrapper\n"
        "from foo import Bar\n"
        "\n"
        "def test_seq():\n"
        '    """Docstring."""\n'
        "\n"
        "    x = cst.parse_module('')\n"
        "    w = MetadataWrapper(x)\n"
        "    b = Bar()\n"
    )
    split = SplitResult(
        new_files={"test_collectors.py": new_file_src},
        original_source="# original\n",
        abort=False,
    )
    placements = [GroupPlacement(group=["test_seq"], target_file="test_collectors.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
    assert vr.verified_functions == 1


def test_verify_inline_import_not_pruned_with_surrounding_blanks_passes():
    # Regression: entity has an inline import surrounded by blank lines that is
    # NOT pruned to a top-level import in the new file.  After
    # _strip_imports_by_line removes the import from the new file's content, the
    # two surrounding blank lines merge into 3+ consecutive newlines before
    # indented code — which _normalize_blank_lines in the new file did NOT
    # collapse (it only runs before the import was stripped in verification).
    # Verification must apply the same _EXCESS_BLANK_BODY_RE normalization to
    # combined_no_imports so the blank-line count matches entity_no_imports.
    post_source = (
        "def test_foo():\n"
        "    x = 1\n"
        "\n"
        "    import pathlib\n"
        "\n"
        "    y = pathlib.Path('.')\n"
        "    return y\n"
    )
    entity = _make_entity("test_foo", 1, 8)
    # New file keeps the inline import (not pruned — no module-level pathlib).
    new_file_src = (
        "def test_foo():\n"
        "    x = 1\n"
        "\n"
        "    import pathlib\n"
        "\n"
        "    y = pathlib.Path('.')\n"
        "    return y\n"
    )
    split = SplitResult(
        new_files={"test_patch.py": new_file_src},
        original_source="# original\n",
        abort=False,
    )
    placements = [GroupPlacement(group=["test_foo"], target_file="test_patch.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
    assert vr.verified_functions == 1


def test_verify_entity_with_name_rewrites_passes():
    # The original entity references SAFE_MODE; after splitting it becomes
    # conversion.SAFE_MODE in the new file.  Verification must apply the
    # name_rewrites before the substring check so it passes rather than
    # reporting a false failure.
    post_source = (
        "def create_runtime(safe_mode=None):\n"
        "    if safe_mode is None:\n"
        "        safe_mode = SAFE_MODE\n"
    )
    entity = _make_entity("create_runtime", 1, 3)
    new_file_src = (
        "def create_runtime(safe_mode=None):\n"
        "    if safe_mode is None:\n"
        "        safe_mode = conversion.SAFE_MODE\n"
    )
    split = SplitResult(
        new_files={"runtime.py": new_file_src},
        original_source="# re-exports\n",
        abort=False,
        entity_name_rewrites={"create_runtime": {"SAFE_MODE": "conversion.SAFE_MODE"}},
    )
    placements = [GroupPlacement(group=["create_runtime"], target_file="runtime.py")]
    vr = _verify_preservation([entity], split, post_source, placements)
    assert vr.failures == []
    # The function passes verification. Only the 1 rewritten line is excluded;
    # the other 2 unchanged lines are credited.
    assert vr.verified_functions == 1
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
