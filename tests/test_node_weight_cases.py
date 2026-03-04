from crispen.refactors.duplicate_extractor import _node_weight
from tests.node_weight_parsing_helpers import _parse_stmt


def test_node_weight_simple_one():
    assert _node_weight(_parse_stmt("a = 1\n")) == 1


def test_node_weight_simple_two_semicolons():
    # Two small stmts on one line separated by semicolon
    stmt = _parse_stmt("a = 1; b = 2\n")
    assert _node_weight(stmt) == 2


def test_node_weight_indented_block():
    block = _parse_stmt("if True:\n    a = 1\n    b = 2\n").body
    assert _node_weight(block) == 2


def test_node_weight_else():
    if_node = _parse_stmt("if True:\n    a = 1\nelse:\n    b = 2\n")
    else_node = if_node.orelse
    assert _node_weight(else_node) == 1


def test_node_weight_finally():
    try_node = _parse_stmt("try:\n    a = 1\nfinally:\n    b = 2\n")
    finally_node = try_node.finalbody
    assert _node_weight(finally_node) == 1


def test_node_weight_functiondef():
    stmt = _parse_stmt("def foo():\n    pass\n")
    assert _node_weight(stmt) == 1
