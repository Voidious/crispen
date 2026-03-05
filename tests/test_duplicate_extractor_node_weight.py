import libcst as cst
from crispen.refactors.duplicate_extractor import _node_weight
from tests.test_duplicate_extractor_ast_utils import _parse_stmt


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


def test_node_weight_classdef():
    stmt = _parse_stmt("class Foo:\n    pass\n")
    assert _node_weight(stmt) == 1


def test_node_weight_non_statement():
    name_node = cst.Name("foo")
    assert _node_weight(name_node) == 0


def test_node_weight_if_no_else():
    # weight = 1 (if) + 2 (body)
    stmt = _parse_stmt("if x:\n    a = 1\n    b = 2\n")
    assert _node_weight(stmt) == 3


def test_node_weight_if_with_else():
    # weight = 1 (if) + 1 (body) + 1 (else body)
    stmt = _parse_stmt("if x:\n    a = 1\nelse:\n    b = 2\n")
    assert _node_weight(stmt) == 3


def test_node_weight_for():
    # weight = 1 (for) + 1 (body)
    stmt = _parse_stmt("for i in x:\n    a = 1\n")
    assert _node_weight(stmt) == 2


def test_node_weight_for_with_else():
    # weight = 1 (for) + 1 (body) + 1 (else body)
    stmt = _parse_stmt("for i in x:\n    a = 1\nelse:\n    b = 2\n")
    assert _node_weight(stmt) == 3


def test_node_weight_while():
    stmt = _parse_stmt("while x:\n    a = 1\n")
    assert _node_weight(stmt) == 2


def test_node_weight_try_with_handler():
    # weight = 1 (try) + 1 (body) + 1 (handler body)
    stmt = _parse_stmt("try:\n    a = 1\nexcept:\n    b = 2\n")
    assert _node_weight(stmt) == 3


def test_node_weight_try_with_handler_and_finally():
    # weight = 1 + 1 + 1 + 1 (finally body)
    stmt = _parse_stmt("try:\n    a = 1\nexcept:\n    b = 2\nfinally:\n    c = 3\n")
    assert _node_weight(stmt) == 4


def test_node_weight_try_with_orelse():
    # weight = 1 + 1 (body) + 1 (handler) + 1 (else body)
    stmt = _parse_stmt("try:\n    a = 1\nexcept:\n    b = 2\nelse:\n    c = 3\n")
    assert _node_weight(stmt) == 4


def test_node_weight_with():
    stmt = _parse_stmt("with open('f') as fh:\n    a = 1\n")
    assert _node_weight(stmt) == 2
