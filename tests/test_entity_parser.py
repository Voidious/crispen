"""Tests for file_limiter.entity_parser — 100% branch coverage."""

from __future__ import annotations

import ast
import textwrap

from crispen.file_limiter.entity_parser import (
    EntityKind,
    _collect_defined_names,
    _find_attached_comment_start,
    _target_names,
    parse_entities,
)


# ---------------------------------------------------------------------------
# _target_names
# ---------------------------------------------------------------------------


def test_target_names_plain_name():
    node = ast.parse("x = 1").body[0].targets[0]
    assert _target_names(node) == ["x"]


def test_target_names_tuple():
    node = ast.parse("(a, b) = 1, 2").body[0].targets[0]
    assert sorted(_target_names(node)) == ["a", "b"]


def test_target_names_nested_tuple():
    node = ast.parse("(a, (b, c)) = 1, (2, 3)").body[0].targets[0]
    assert sorted(_target_names(node)) == ["a", "b", "c"]


def test_target_names_list_target():
    node = ast.parse("[a, b] = [1, 2]").body[0].targets[0]
    assert sorted(_target_names(node)) == ["a", "b"]


def test_target_names_attribute_returns_empty():
    node = ast.parse("obj.attr = 1").body[0].targets[0]
    assert _target_names(node) == []


# ---------------------------------------------------------------------------
# _collect_defined_names
# ---------------------------------------------------------------------------


def _first_stmt(source: str) -> ast.AST:
    return ast.parse(source).body[0]


def test_collect_function_def():
    assert _collect_defined_names(_first_stmt("def foo(): pass")) == ["foo"]


def test_collect_async_function_def():
    assert _collect_defined_names(_first_stmt("async def bar(): pass")) == ["bar"]


def test_collect_class_def():
    assert _collect_defined_names(_first_stmt("class Baz: pass")) == ["Baz"]


def test_collect_assign_single():
    assert _collect_defined_names(_first_stmt("x = 1")) == ["x"]


def test_collect_assign_multiple_targets():
    # x = y = 1 has two targets
    names = _collect_defined_names(_first_stmt("x = y = 1"))
    assert sorted(names) == ["x", "y"]


def test_collect_ann_assign_with_value():
    assert _collect_defined_names(_first_stmt("x: int = 5")) == ["x"]


def test_collect_ann_assign_no_value():
    # Bare annotation without assignment does not define a name
    assert _collect_defined_names(_first_stmt("x: int")) == []


def test_collect_ann_assign_non_name_target():
    # Annotated attribute assignment — target is Attribute, not Name
    assert _collect_defined_names(_first_stmt("Foo.attr: int = 5")) == []


def test_collect_import_simple():
    assert sorted(_collect_defined_names(_first_stmt("import os, sys"))) == [
        "os",
        "sys",
    ]


def test_collect_import_with_asname():
    assert _collect_defined_names(_first_stmt("import os as operating_system")) == [
        "operating_system"
    ]


def test_collect_import_dotted_uses_first_component():
    assert _collect_defined_names(_first_stmt("import os.path")) == ["os"]


def test_collect_import_from():
    names = _collect_defined_names(_first_stmt("from os import path, getcwd"))
    assert sorted(names) == ["getcwd", "path"]


def test_collect_import_from_with_asname():
    assert _collect_defined_names(_first_stmt("from os import path as p")) == ["p"]


def test_collect_other_returns_empty():
    # del statement — not a definition
    assert _collect_defined_names(_first_stmt("del x")) == []


# ---------------------------------------------------------------------------
# _find_attached_comment_start
# ---------------------------------------------------------------------------


def test_find_attached_comment_start_at_file_start():
    lines = ["def foo():\n", "    pass\n"]
    # stmt_start = 1; i starts at -1 → loop never runs
    assert _find_attached_comment_start(lines, 1) == 1


def test_find_attached_comment_start_no_preceding_comment():
    lines = ["x = 1\n", "def foo():\n", "    pass\n"]
    # line before stmt (0-indexed index 0) is "x = 1", not a comment
    assert _find_attached_comment_start(lines, 2) == 2


def test_find_attached_comment_start_one_comment():
    lines = ["# attached\n", "def foo():\n", "    pass\n"]
    assert _find_attached_comment_start(lines, 2) == 1


def test_find_attached_comment_start_multiple_comments():
    lines = ["# line 1\n", "# line 2\n", "def foo():\n"]
    assert _find_attached_comment_start(lines, 3) == 1


def test_find_attached_comment_start_blank_line_stops():
    lines = ["# unattached\n", "\n", "# attached\n", "def foo():\n"]
    # blank line at index 1 stops the scan
    assert _find_attached_comment_start(lines, 4) == 3


# ---------------------------------------------------------------------------
# parse_entities
# ---------------------------------------------------------------------------


def test_parse_empty_module():
    assert parse_entities("") == []


def test_parse_syntax_error():
    assert parse_entities("def (invalid") == []


def test_parse_single_function():
    source = "def foo():\n    pass\n"
    entities = parse_entities(source)
    assert len(entities) == 1
    e = entities[0]
    assert e.kind == EntityKind.FUNCTION
    assert e.name == "foo"
    assert e.start_line == 1
    assert e.end_line == 2
    assert e.names_defined == ["foo"]


def test_parse_async_function():
    source = "async def bar():\n    pass\n"
    entities = parse_entities(source)
    assert len(entities) == 1
    e = entities[0]
    assert e.kind == EntityKind.FUNCTION
    assert e.name == "bar"


def test_parse_single_class():
    source = "class Foo:\n    pass\n"
    entities = parse_entities(source)
    assert len(entities) == 1
    e = entities[0]
    assert e.kind == EntityKind.CLASS
    assert e.name == "Foo"
    assert e.names_defined == ["Foo"]


def test_parse_top_level_assignment():
    entities = parse_entities("X = 1\n")
    assert len(entities) == 1
    e = entities[0]
    assert e.kind == EntityKind.TOP_LEVEL
    assert e.name == "_block_1"
    assert e.names_defined == ["X"]
    assert e.start_line == 1
    assert e.end_line == 1


def test_parse_imports_form_top_level_entity():
    entities = parse_entities("import os\nfrom sys import argv\n")
    assert len(entities) == 1
    e = entities[0]
    assert e.kind == EntityKind.TOP_LEVEL
    assert "os" in e.names_defined
    assert "argv" in e.names_defined


def test_parse_multiple_entities_order():
    source = textwrap.dedent(
        """\
        import os
        X = 1
        def foo():
            pass
        class Bar:
            pass
    """
    )
    entities = parse_entities(source)
    kinds = [e.kind for e in entities]
    names = [e.name for e in entities]
    assert EntityKind.TOP_LEVEL in kinds
    assert EntityKind.FUNCTION in kinds
    assert EntityKind.CLASS in kinds
    assert "foo" in names
    assert "Bar" in names
    # Source order is preserved
    top = next(e for e in entities if e.kind == EntityKind.TOP_LEVEL)
    foo = next(e for e in entities if e.name == "foo")
    bar = next(e for e in entities if e.name == "Bar")
    assert top.start_line < foo.start_line < bar.start_line


def test_parse_attached_comment_included_in_entity():
    source = "# Comment for foo\ndef foo():\n    pass\n"
    entities = parse_entities(source)
    assert len(entities) == 1
    assert entities[0].start_line == 1  # comment line is included


def test_parse_comment_separated_by_blank_not_attached():
    source = "# Standalone\n\ndef foo():\n    pass\n"
    entities = parse_entities(source)
    foo = next(e for e in entities if e.name == "foo")
    # blank line separates comment from def → comment not attached
    assert foo.start_line == 3


def test_parse_decorated_function_uses_decorator_line():
    source = "# Comment\n@decorator\ndef foo():\n    pass\n"
    entities = parse_entities(source)
    assert len(entities) == 1
    e = entities[0]
    # comment is attached to decorator, so start at line 1
    assert e.start_line == 1
    assert e.name == "foo"
    assert e.kind == EntityKind.FUNCTION


def test_parse_decorated_class():
    source = "@decorator\nclass Foo:\n    pass\n"
    entities = parse_entities(source)
    assert len(entities) == 1
    e = entities[0]
    assert e.start_line == 1
    assert e.kind == EntityKind.CLASS


def test_parse_undecorated_function_no_extra_comment():
    # Function at line 1 with no preceding comment
    source = "def foo():\n    pass\n"
    entities = parse_entities(source)
    assert entities[0].start_line == 1


def test_parse_top_level_blocks_split_by_function():
    source = "X = 1\ndef foo():\n    pass\nY = 2\n"
    entities = parse_entities(source)
    names = [e.name for e in entities]
    assert "_block_1" in names  # X = 1
    assert "foo" in names
    assert "_block_4" in names  # Y = 2


def test_parse_flush_block_empty_at_loop_start():
    # Function is the very first statement; _flush_block sees empty pending_block
    source = "def foo():\n    pass\n"
    entities = parse_entities(source)
    assert [e.name for e in entities] == ["foo"]


def test_parse_top_level_block_names_defined():
    # Multiple statements in one block; all names should be collected
    source = "import os\nX = 1\na, b = 2, 3\ndef foo():\n    pass\n"
    entities = parse_entities(source)
    block = next(e for e in entities if e.kind == EntityKind.TOP_LEVEL)
    assert "os" in block.names_defined
    assert "X" in block.names_defined
    assert "a" in block.names_defined
    assert "b" in block.names_defined
