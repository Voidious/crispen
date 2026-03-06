import textwrap
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    _SeqInfo,
    _SequenceCollector,
    _filter_maximal_groups,
    _find_duplicate_groups,
    _has_def,
    _node_weight,
    _normalize_source,
    _overlaps_diff,
    _sequence_weight,
    _seq_ends_with_return,
    _replacement_contains_return,
)
from .test_edit_and_insertion_utils import _make_seq_with_source


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


def test_normalize_source_normalizes_vars():
    src = "result = compute(data)\noutput = transform(result)\n"
    norm = _normalize_source(src)
    # All names (both assigned and free) are replaced with positional placeholders
    assert "result" not in norm
    assert "output" not in norm
    assert "compute" not in norm
    assert "data" not in norm


def test_normalize_source_same_fingerprint():
    src_a = "x = compute(data)\ny = transform(x)\n"
    src_b = "val = compute(data)\nres = transform(val)\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_different_ops():
    # Structurally different code (different number of statements) should differ
    src_a = "x = a + b\n"
    src_b = "x = a + b\ny = x * 2\n"
    assert _normalize_source(src_a) != _normalize_source(src_b)


def test_normalize_source_invalid_syntax():
    src = "def f(: pass"
    # Falls back to original source
    assert _normalize_source(src) == src


def test_normalize_source_load_context_replaced():
    # Var assigned then used: both should be normalized the same
    src_a = "x = 1\ny = x + 1\n"
    src_b = "a = 1\nb = a + 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_load_not_in_map():
    # Free variables (Load context, never stored) are also normalized,
    # so two blocks with different free variable names get the same fingerprint.
    src_a = "y = a + 1\n"
    src_b = "z = b + 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_repeated_store():
    # Same name assigned twice: _placeholder called with cached key (False branch)
    src = "x = 1\nx = 2\n"
    norm = _normalize_source(src)
    # Both assignments normalize to the same placeholder
    assert norm.count("_v0") == 2


def test_normalize_source_del_context():
    # Del context falls through to return node unchanged
    src = "del x\n"
    norm = _normalize_source(src)
    assert "x" in norm


def test_normalize_source_free_variables_match():
    # Blocks differing only in free variable names should get the same fingerprint.
    # This is the core case: `p = a * 2; if p > 100: p += 1` vs the same with q/b.
    src_a = "p = a * 2\nif p > 100:\n    p += 1\n"
    src_b = "q = b * 2\nif q > 100:\n    q += 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_indented_blocks_match():
    # Source collected from inside a function is indented; dedent must happen
    # before ast.parse so that structurally identical blocks still match.
    src_a = "    p = a * 2\n    if p > 100:\n        p += 1\n"
    src_b = "    q = b * 2\n    if q > 100:\n        q += 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


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


def _collect_sequences(source: str, max_seq_len: int = 8):
    tree = cst.parse_module(source)
    lines = source.splitlines(keepends=True)
    collector = _SequenceCollector(lines, max_seq_len=max_seq_len)
    MetadataWrapper(tree).visit(collector)
    return collector.sequences


def test_collector_finds_sequences():
    source = textwrap.dedent(
        """\
        def foo():
            a = 1
            b = 2
            c = 3
        """
    )
    seqs = _collect_sequences(source)
    assert len(seqs) > 0


def test_collector_skips_light_sequences():
    # Only 2 statements — below weight threshold of 3
    source = textwrap.dedent(
        """\
        def foo():
            a = 1
            b = 2
        """
    )
    seqs = _collect_sequences(source)
    assert all(seq.start_line != seq.end_line or len(seq.stmts) >= 2 for seq in seqs)
    # All 2-stmt windows skipped because weight < 3
    assert len([s for s in seqs if len(s.stmts) == 2]) == 0


def test_collector_skips_defs():
    source = textwrap.dedent(
        """\
        def foo():
            pass
        def bar():
            pass
        def baz():
            pass
        """
    )
    seqs = _collect_sequences(source)
    # Module-level sequences of defs should be skipped
    for seq in seqs:
        assert not _has_def(seq.stmts)


def test_collector_scope_tracking():
    source = textwrap.dedent(
        """\
        def my_func():
            a = 1
            b = 2
            c = 3
        """
    )
    seqs = _collect_sequences(source)
    func_seqs = [s for s in seqs if s.scope == "my_func"]
    assert len(func_seqs) > 0


def test_sequence_collector_custom_max_seq_len():
    # max_seq_len=2 means windows are at most 2 statements.
    # With 4 statements each of weight 1, all 2-stmt windows have weight 2 <
    # MIN_WEIGHT=3.  So no sequences pass the weight filter → sequences == [].
    source = textwrap.dedent(
        """\
        def foo():
            a = 1
            b = 2
            c = 3
            d = 4
        """
    )
    seqs = _collect_sequences(source, max_seq_len=2)
    # No 3-stmt (or larger) windows generated; all ≤2-stmt windows fail weight check.
    assert all(len(s.stmts) <= 2 for s in seqs)
    assert seqs == []


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


def test_replacement_contains_return_true():
    assert _replacement_contains_return("    return x\n") is True


def test_replacement_contains_return_false():
    assert _replacement_contains_return("    _helper()\n") is False


def test_replacement_contains_return_syntax_error():
    # Unclosed paren inside the wrapper → SyntaxError → False.
    assert _replacement_contains_return("    (\n") is False
