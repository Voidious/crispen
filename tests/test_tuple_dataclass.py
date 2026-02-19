"""Tests for the TupleDataclass refactor."""

import libcst as cst
from libcst.metadata import MetadataWrapper

from crispen.refactors.tuple_dataclass import TupleDataclass


def _apply(source: str, ranges=None, min_size: int = 3) -> str:
    if ranges is None:
        ranges = [(1, 100)]
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    transformer = TupleDataclass(ranges, min_size=min_size)
    new_tree = wrapper.visit(transformer)
    return new_tree.code


def _changes(source: str, ranges=None, min_size: int = 3) -> list:
    if ranges is None:
        ranges = [(1, 100)]
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    transformer = TupleDataclass(ranges, min_size=min_size)
    wrapper.visit(transformer)
    return transformer.changes_made


# ---------------------------------------------------------------------------
# Happy path: simple tuple in module scope
# ---------------------------------------------------------------------------

BEFORE_SIMPLE = "result = (1, 2, 3)\n"


def test_simple_tuple_transformed():
    result = _apply(BEFORE_SIMPLE)
    assert "DataTuple(" in result or "Tuple(" in result
    assert "field_0" in result
    assert "field_1" in result
    assert "field_2" in result


def test_simple_tuple_imports_added():
    result = _apply(BEFORE_SIMPLE)
    assert "from dataclasses import dataclass" in result
    assert "from typing import Any" in result


def test_simple_tuple_class_defined():
    result = _apply(BEFORE_SIMPLE)
    assert "@dataclass" in result
    assert "class " in result
    assert "field_0: Any" in result


def test_reports_change():
    changes = _changes(BEFORE_SIMPLE)
    assert len(changes) == 1
    assert "TupleDataclass" in changes[0]


# ---------------------------------------------------------------------------
# Skip cases
# ---------------------------------------------------------------------------

BEFORE_SMALL = "result = (1, 2)\n"


def test_small_tuple_skipped():
    result = _apply(BEFORE_SMALL)
    assert result == BEFORE_SMALL


def test_small_tuple_no_change():
    assert _changes(BEFORE_SMALL) == []


BEFORE_STARRED = "result = (*a, 1, 2, 3)\n"


def test_starred_skipped():
    result = _apply(BEFORE_STARRED)
    assert result == BEFORE_STARRED


# ---------------------------------------------------------------------------
# min_size parameter
# ---------------------------------------------------------------------------


def test_min_size_two():
    source = "x = (1, 2)\n"
    result = _apply(source, min_size=2)
    assert "field_0" in result
    assert "field_1" in result


def test_min_size_four_skips_three():
    source = "x = (1, 2, 3)\n"
    result = _apply(source, min_size=4)
    assert result == source


# ---------------------------------------------------------------------------
# Range filtering
# ---------------------------------------------------------------------------


def test_outside_range_skipped():
    result = _apply(BEFORE_SIMPLE, ranges=[(50, 100)])
    assert result == BEFORE_SIMPLE


# ---------------------------------------------------------------------------
# Output is valid Python
# ---------------------------------------------------------------------------


def test_output_is_valid_python():
    result = _apply(BEFORE_SIMPLE)
    # Should not raise
    compile(result, "<string>", "exec")


# ---------------------------------------------------------------------------
# Doesn't duplicate import if already present
# ---------------------------------------------------------------------------

BEFORE_WITH_IMPORTS = """\
from dataclasses import dataclass
from typing import Any

result = (1, 2, 3)
"""


def test_no_duplicate_imports():
    result = _apply(BEFORE_WITH_IMPORTS)
    assert result.count("from dataclasses import dataclass") == 1
    assert result.count("from typing import Any") == 1
