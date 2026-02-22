"""Tests for the IfNotElse refactor."""

import libcst as cst
from libcst.metadata import MetadataWrapper

from crispen.refactors.if_not_else import IfNotElse


def _create_transformer(source: str, ranges):
    if ranges is None:
        ranges = [(1, 100)]
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    transformer = IfNotElse(ranges)
    return wrapper, transformer


def _apply(source: str, ranges=None) -> str:
    wrapper, transformer = _create_transformer(source, ranges)
    new_tree = wrapper.visit(transformer)
    return new_tree.code


def _changes(source: str, ranges=None) -> list:
    wrapper, transformer = _create_transformer(source, ranges)
    wrapper.visit(transformer)
    return transformer.changes_made


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

BEFORE_SIMPLE = """\
if not x:
    a()
else:
    b()
"""

AFTER_SIMPLE = """\
if x:
    b()
else:
    a()
"""


def test_simple_flip():
    assert _apply(BEFORE_SIMPLE) == AFTER_SIMPLE


def test_reports_change():
    changes = _changes(BEFORE_SIMPLE)
    assert len(changes) == 1
    assert "IfNotElse" in changes[0]


BEFORE_COMPLEX_CONDITION = """\
if not (a and b):
    do_x()
else:
    do_y()
"""

AFTER_COMPLEX_CONDITION = """\
if (a and b):
    do_y()
else:
    do_x()
"""


def test_complex_condition():
    assert _apply(BEFORE_COMPLEX_CONDITION) == AFTER_COMPLEX_CONDITION


# ---------------------------------------------------------------------------
# Skip cases
# ---------------------------------------------------------------------------

NO_ELSE = """\
if not x:
    a()
"""


def test_no_else_skipped():
    result = _apply(NO_ELSE)
    assert result == NO_ELSE


def test_no_else_no_changes():
    assert _changes(NO_ELSE) == []


ELIF_CHAIN = """\
if not x:
    a()
elif y:
    b()
else:
    c()
"""


def test_elif_skipped():
    result = _apply(ELIF_CHAIN)
    assert result == ELIF_CHAIN


NO_NOT = """\
if x:
    a()
else:
    b()
"""


def test_no_not_skipped():
    result = _apply(NO_NOT)
    assert result == NO_NOT


# ---------------------------------------------------------------------------
# Range filtering
# ---------------------------------------------------------------------------


def test_outside_range_skipped():
    # The if statement is on line 1, but we only have range starting at line 10
    result = _apply(BEFORE_SIMPLE, ranges=[(10, 20)])
    assert result == BEFORE_SIMPLE


def test_inside_range_transformed():
    result = _apply(BEFORE_SIMPLE, ranges=[(1, 1)])
    assert result == AFTER_SIMPLE


# ---------------------------------------------------------------------------
# Preserves surrounding code
# ---------------------------------------------------------------------------

BEFORE_SURROUNDED = """\
x = 1
if not flag:
    do_a()
else:
    do_b()
y = 2
"""

AFTER_SURROUNDED = """\
x = 1
if flag:
    do_b()
else:
    do_a()
y = 2
"""


def test_preserves_surrounding():
    assert _apply(BEFORE_SURROUNDED) == AFTER_SURROUNDED


# ---------------------------------------------------------------------------
# Non-Not unary operator is skipped
# ---------------------------------------------------------------------------

BITWISE_NOT = """\
if ~x:
    a()
else:
    b()
"""


def test_bitwise_not_skipped():
    result = _apply(BITWISE_NOT)
    assert result == BITWISE_NOT
