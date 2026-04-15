from unittest.mock import MagicMock, patch
import textwrap
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _SeqInfo,
    _SequenceCollector,
    _filter_maximal_groups,
    _find_duplicate_groups,
    _has_def,
    _has_internal_overlap,
    _has_param_overwritten_before_read,
    _is_pure_literal,
    _node_weight,
    _overlaps_diff,
    _replace_unused_in_target,
    _scope_end_line,
    _seq_ends_with_return,
    _seq_source_contains_yield,
    _sequence_weight,
    _strip_unused_call_assignments,
)
import libcst as cst
from .test_duplicate_extractor import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from .test_replacement_utils import _make_seq_with_source
from .test_sequence_collector import _collect_sequences


def _parse_stmt(src: str) -> cst.BaseStatement:
    return cst.parse_module(src).body[0]


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


def test_sequence_weight_empty():
    assert _sequence_weight([]) == 0


def test_sequence_weight_mixed():
    stmts = [
        _parse_stmt("a = 1\n"),
        _parse_stmt("if x:\n    b = 2\n"),
    ]
    assert _sequence_weight(stmts) == 1 + 2


def test_has_def_no_def():
    stmts = [_parse_stmt("a = 1\n"), _parse_stmt("b = 2\n")]
    assert _has_def(stmts) is False


def test_has_def_with_functiondef():
    stmts = [_parse_stmt("a = 1\n"), _parse_stmt("def foo():\n    pass\n")]
    assert _has_def(stmts) is True


def test_has_def_with_classdef():
    stmts = [_parse_stmt("class Foo:\n    pass\n")]
    assert _has_def(stmts) is True


def _make_seq(start: int, end: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="<module>",
        source="",
        fingerprint="",
    )


def test_overlaps_diff_yes():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(8, 12)]) is True


def test_overlaps_diff_no():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(11, 20)]) is False


def test_overlaps_diff_exact_boundary():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(10, 15)]) is True


def test_find_duplicate_groups_empty():
    assert _find_duplicate_groups([], [(1, 5)]) == []


def test_find_duplicate_groups_singleton():
    seq = _make_seq(1, 3)
    seq.fingerprint = "fp1"
    seqs = [seq]
    # Only one seq with this fingerprint — not a duplicate
    assert _find_duplicate_groups(seqs, [(1, 3)]) == []


def test_find_duplicate_groups_no_diff_overlap():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    # Neither overlaps diff range (20, 30)
    assert _find_duplicate_groups([s1, s2], [(20, 30)]) == []


def test_find_duplicate_groups_valid():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    groups = _find_duplicate_groups([s1, s2], [(1, 3)])
    assert len(groups) == 1
    assert set(id(s) for s in groups[0]) == {id(s1), id(s2)}


def test_has_internal_overlap_no_overlap():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    assert not _has_internal_overlap([s1, s2])


def test_has_internal_overlap_adjacent_no_overlap():
    # end_line of s1 == start_line - 1 of s2: not overlapping
    s1 = _SeqInfo([], 1, 5, "<module>", "", "fp1")
    s2 = _SeqInfo([], 6, 10, "<module>", "", "fp1")
    assert not _has_internal_overlap([s1, s2])


def test_has_internal_overlap_touching():
    # end_line of s1 == start_line of s2: overlap (shared boundary line)
    s1 = _SeqInfo([], 1, 5, "<module>", "", "fp1")
    s2 = _SeqInfo([], 5, 9, "<module>", "", "fp1")
    assert _has_internal_overlap([s1, s2])


def test_has_internal_overlap_proper_overlap():
    s1 = _SeqInfo([], 27, 30, "<module>", "", "fp1")
    s2 = _SeqInfo([], 29, 32, "<module>", "", "fp1")
    assert _has_internal_overlap([s1, s2])


def test_has_internal_overlap_unsorted_order():
    # Sequences given in reverse order — function must sort before checking.
    s1 = _SeqInfo([], 29, 32, "<module>", "", "fp1")
    s2 = _SeqInfo([], 27, 30, "<module>", "", "fp1")
    assert _has_internal_overlap([s1, s2])


def test_find_duplicate_groups_skips_internally_overlapping():
    # Simulate the op_range pattern: two pairs [A,B] and [B,C] that share a
    # statement.  The group has internal overlap and must be filtered out.
    s1 = _SeqInfo([], 27, 30, "<module>", "", "fp1")
    s2 = _SeqInfo([], 29, 32, "<module>", "", "fp1")
    # Diff covers both sequences.
    groups = _find_duplicate_groups([s1, s2], [(27, 32)])
    assert groups == []


def test_find_duplicate_groups_caps_at_max_groups():
    sequences = []
    for i in range(6):
        fp = f"fp{i}"
        # Place each group in a disjoint band of 20 lines so _filter_maximal_groups
        # keeps all 6 (none overlap), and the max_groups=3 cap is what limits output.
        sequences.append(_SeqInfo([], i * 20 + 1, i * 20 + 3, "<module>", "", fp))
        sequences.append(_SeqInfo([], i * 20 + 10, i * 20 + 12, "<module>", "", fp))
    # Diff range covers all sequences so the diff-overlap filter passes for all.
    groups = _find_duplicate_groups(sequences, [(1, 130)], max_groups=3)
    assert len(groups) == 3


def test_filter_maximal_groups_empty():
    assert _filter_maximal_groups([]) == []


def test_filter_maximal_groups_single_group():
    s1 = _SeqInfo([], 1, 10, "<module>", "", "fp1")
    s2 = _SeqInfo([], 20, 29, "<module>", "", "fp1")
    group = [s1, s2]
    result = _filter_maximal_groups([group])
    assert result == [group]


def test_filter_maximal_groups_removes_subsumed():
    # Large group spans lines 1-10; small group spans 1-5 (subset).
    # Only the large group should be kept.
    s_large_a = _SeqInfo([], 1, 10, "<module>", "", "fp_large")
    s_large_b = _SeqInfo([], 20, 29, "<module>", "", "fp_large")
    large_group = [s_large_a, s_large_b]

    s_small_a = _SeqInfo([], 1, 5, "<module>", "", "fp_small")
    s_small_b = _SeqInfo([], 20, 24, "<module>", "", "fp_small")
    small_group = [s_small_a, s_small_b]

    result = _filter_maximal_groups([small_group, large_group])
    assert len(result) == 1
    assert result[0] is large_group


def test_filter_maximal_groups_keeps_non_overlapping():
    # Two groups with completely disjoint line ranges — both should be kept.
    s1a = _SeqInfo([], 1, 5, "<module>", "", "fp1")
    s1b = _SeqInfo([], 30, 34, "<module>", "", "fp1")
    group1 = [s1a, s1b]

    s2a = _SeqInfo([], 10, 14, "<module>", "", "fp2")
    s2b = _SeqInfo([], 40, 44, "<module>", "", "fp2")
    group2 = [s2a, s2b]

    result = _filter_maximal_groups([group1, group2])
    assert len(result) == 2


def test_has_param_overwritten_before_read_false_when_param_is_read():
    # Parameter is read before (or without) being reassigned — should return False.
    helper = "def fn(x):\n    return x + 1\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_has_param_overwritten_before_read_true_when_immediately_overwritten():
    # Parameter is assigned on the first statement without being read — True.
    helper = "def setup(client):\n    client = object()\n    return client\n"
    assert _has_param_overwritten_before_read(helper) is True


def test_has_param_overwritten_before_read_false_for_conditional_default():
    # The ``if x is None: x = default`` pattern reads before writing — False.
    helper = "def fn(x=None):\n    if x is None:\n        x = []\n    return x\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_has_param_overwritten_before_read_vararg_and_kwarg():
    # Covers the vararg/kwarg branches — neither is overwritten here.
    helper = "def fn(*args, **kwargs):\n    return args, kwargs\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_is_pure_literal_constant():
    import ast

    assert _is_pure_literal(ast.parse("0", mode="eval").body)
    assert _is_pure_literal(ast.parse('"s"', mode="eval").body)
    assert _is_pure_literal(ast.parse("None", mode="eval").body)
    assert _is_pure_literal(ast.parse("True", mode="eval").body)


def test_is_pure_literal_containers():
    import ast

    assert _is_pure_literal(ast.parse("[]", mode="eval").body)
    assert _is_pure_literal(ast.parse("(1, 2)", mode="eval").body)
    assert _is_pure_literal(ast.parse("{1: 2}", mode="eval").body)
    assert _is_pure_literal(ast.parse("{1, 2}", mode="eval").body)


def test_is_pure_literal_call_is_false():
    import ast

    assert not _is_pure_literal(ast.parse("func()", mode="eval").body)


def test_is_pure_literal_name_is_false():
    import ast

    assert not _is_pure_literal(ast.parse("x", mode="eval").body)


def test_is_pure_literal_nested_call_is_false():
    import ast

    assert not _is_pure_literal(ast.parse("[func()]", mode="eval").body)


def _make_source_lines(src: str):
    return src.splitlines(keepends=True)


def test_scope_end_line_module_returns_full_length():
    lines = _make_source_lines("x = 1\ny = 2\n")
    assert _scope_end_line(lines, "<module>", 1) == len(lines)


def test_scope_end_line_function_scope():
    src = "def foo():\n    x = 1\n    y = 2\n\ndef bar():\n    z = 3\n"
    lines = _make_source_lines(src)
    # Block ends at line 2 (inside foo). foo ends at line 3.
    assert _scope_end_line(lines, "foo", 2) == 3


def test_scope_end_line_does_not_bleed_into_next_function():
    src = "def foo():\n    x = 1\n\ndef bar():\n    x = 2\n"
    lines = _make_source_lines(src)
    # Searching for `x` after line 2 should stop at end of foo (line 2), not
    # reach bar where `x` also appears.
    end = _scope_end_line(lines, "foo", 2)
    assert end == 2  # foo ends at line 2; bar's x is excluded


def test_scope_end_line_picks_innermost_matching_scope():
    # Two functions named "inner" — one nested inside outer, one at module level.
    src = (
        "def outer():\n"
        "    def inner():\n"
        "        a = 1\n"
        "    inner()\n"
        "\n"
        "def inner():\n"
        "    b = 2\n"
    )
    lines = _make_source_lines(src)
    # Block at line 3 is inside the nested inner (lines 2-3). That is the
    # smallest matching span, so end_lineno == 3 is returned.
    assert _scope_end_line(lines, "inner", 3) == 3


def test_scope_end_line_class_scope():
    src = "class Foo:\n    x = 1\n    y = 2\n\nclass Bar:\n    x = 3\n"
    lines = _make_source_lines(src)
    assert _scope_end_line(lines, "Foo", 2) == 3


def test_scope_end_line_no_match_returns_full_length():
    src = "def foo():\n    x = 1\n"
    lines = _make_source_lines(src)
    # Scope name doesn't match any definition.
    assert _scope_end_line(lines, "bar", 1) == len(lines)


def test_scope_end_line_syntax_error_returns_full_length():
    lines = _make_source_lines("def (\n    x = 1\n")
    assert _scope_end_line(lines, "foo", 1) == len(lines)


def test_replace_unused_in_target_name_used():
    import ast

    target = ast.parse("result = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(result)\n")
    assert all_r is False and any_r is False
    assert ast.unparse(new_t) == "result"


def test_replace_unused_in_target_name_unused():
    import ast

    target = ast.parse("result = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is True and any_r is True
    assert ast.unparse(new_t) == "_"


def test_replace_unused_in_target_tuple_all_unused():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is True and any_r is True
    assert ast.unparse(new_t) == "(_, _)"


def test_replace_unused_in_target_tuple_some_unused():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(a)\n")
    assert all_r is False and any_r is True
    assert ast.unparse(new_t) == "(a, _)"


def test_replace_unused_in_target_tuple_all_used():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(a, b)\n")
    assert all_r is False and any_r is False


def test_replace_unused_in_target_attribute_treated_as_used():
    import ast

    target = ast.parse("self.x = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is False and any_r is False


def test_strip_unused_call_assignments_removes_unused_single():
    # `result` never appears after the block → assignment stripped.
    replacement = "    result = _helper(x, y)\n"
    following = ["    do_something()\n", "    return z\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x, y)\n"


def test_strip_unused_call_assignments_keeps_used_single():
    # `result` is referenced after the block → assignment kept.
    replacement = "    result = _helper(x, y)\n"
    following = ["    print(result)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_removes_unused_tuple():
    # Both names unused after the block → assignment stripped entirely.
    replacement = "    a, b = _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n"


def test_strip_unused_call_assignments_partial_tuple_replaces_with_underscore():
    # One name used, one unused → replace unused with _.
    replacement = "    a, b = _helper(x)\n"
    following = ["    print(a)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    (a, _) = _helper(x)\n"


def test_strip_unused_call_assignments_attribute_target_unchanged():
    # Target is an attribute (self.x = call()) → treated as used → left unchanged.
    replacement = "    self.result = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_non_call_rhs_unchanged():
    # RHS is not a Call → leave unchanged.
    replacement = "    result = x + y\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_chained_all_unused_stripped():
    # Chained assignment where every name is unused → stripped to just the call.
    replacement = "    a = b = _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n"


def test_restrip_drops_assignment_unused_only_after_all_call_sites_replaced(
    monkeypatch,
):
    # Regression: when two call sites reference the same variable name, the
    # per-call-site strip (which uses original following lines) sees the name
    # in the other call site's original block and keeps the assignment.  After
    # all replacements are assembled the variable is truly unused, so the
    # re-strip pass must drop it.
    #
    # Source:  test_f has two identical 2-line blocks.
    # LLM returns:
    #   - call site 1 replacement: ``data = assert_error(result)``
    #   - call site 2 replacement: ``assert_error(result2)``   (no assignment)
    # After initial per-call-site strip, call site 1 keeps the assignment
    # because "data" appears in the original following source (inside call
    # site 2's original block).  The re-strip must then drop it.
    # Using function parameters avoids the SequenceCollector merging the
    # assignment lines into the duplicate block.
    # Use 3-statement blocks (weight=3 ≥ min_weight) so the SequenceCollector
    # finds the duplicate group.  Mirroring the real lever-mcp pattern:
    # json.loads + two asserts.  Both result and result2 are function
    # parameters so the SequenceCollector cannot absorb the assignment lines
    # into the duplicate block.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        def test_f(result, result2):
            rd = json.loads(result)
            assert rd["value"] is None
            assert "error" in rd
            rd = json.loads(result2)
            assert rd["value"] is None
            assert "error" in rd
        """
    )
    helper = textwrap.dedent(
        """\
        def assert_error_result(result):
            rd = json.loads(result)
            assert rd["value"] is None
            assert "error" in rd
        """
    )
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "identical blocks"),
            _make_extract_response(
                {
                    "function_name": "assert_error_result",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        # LLM assigns the return value at call site 1 …
                        "    rd = assert_error_result(result)\n",
                        # … but not at call site 2 (helper returns None).
                        "    assert_error_result(result2)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor([(2, 4), (5, 7)], source=source)

    assert de._new_source is not None
    # The re-strip must have dropped the unused assignment at call site 1.
    assert "rd = assert_error_result(result)" not in de._new_source
    assert "assert_error_result(result)" in de._new_source
    assert "assert_error_result(result2)" in de._new_source


def test_sequence_collector_min_weight_filters_light_sequences():
    # A single assignment has weight 1. With min_weight=2 it should be excluded.
    source = "def f():\n    a = 1\n    b = 2\n"
    source_lines = source.splitlines(keepends=True)
    tree = cst.parse_module(source)

    collector = _SequenceCollector(source_lines, max_seq_len=2, min_weight=2)
    MetadataWrapper(tree).visit(collector)
    # Single-statement sequences (weight=1) should be filtered out
    single_stmt_seqs = [s for s in collector.sequences if len(s.stmts) == 1]
    assert single_stmt_seqs == []


def test_seq_ends_with_return_true():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return x\n"))
        is True
    )


def test_seq_ends_with_return_false_no_return():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    y = 2\n")) is False
    )


def test_seq_ends_with_return_syntax_error():
    assert _seq_ends_with_return(_make_seq_with_source("    (\n")) is False


def test_seq_ends_with_return_empty_body():
    # Pure whitespace → ast.parse produces an empty module body.
    assert _seq_ends_with_return(_make_seq_with_source("   \n")) is False


def test_seq_ends_with_return_bare_return():
    # Bare `return` is equivalent to returning None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return\n")) is False
    )


def test_seq_ends_with_return_return_none():
    # Explicit `return None` is also equivalent to implicit None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return None\n"))
        is False
    )


def test_seq_source_contains_yield_async_with_yield():
    # The exact pattern that triggered the bug: async with ... as c: yield c
    src = "    async with Client(mcp) as c:\n        yield c\n"
    assert _seq_source_contains_yield(src) is True


def test_seq_source_contains_yield_plain_yield():
    assert _seq_source_contains_yield("    yield x\n") is True


def test_seq_source_contains_yield_from():
    assert _seq_source_contains_yield("    yield from something()\n") is True


def test_seq_source_contains_yield_no_yield():
    assert _seq_source_contains_yield("    x = 1\n    y = 2\n") is False


def test_seq_source_contains_yield_nested_funcdef_not_counted():
    # yield inside a nested def must NOT trigger the guard
    src = "    def inner():\n        yield 1\n"
    assert _seq_source_contains_yield(src) is False


def test_seq_source_contains_yield_syntax_error():
    assert _seq_source_contains_yield("    (\n") is False


def test_collector_skips_yield_sequences():
    # Sequences whose source contains yield should never be collected.
    source = textwrap.dedent(
        """\
        async def make_client():
            x = setup()
            async with Client(x) as c:
                yield c

        async def make_client2():
            x = setup()
            async with Client(x) as c:
                yield c
        """
    )
    seqs = _collect_sequences(source)
    for seq in seqs:
        assert not _seq_source_contains_yield(seq.source)
