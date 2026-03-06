import textwrap
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    _extract_defined_names,
    _FunctionCollector,
    _SeqInfo,
    _SequenceCollector,
    _has_def,
    _find_escaping_vars,
    _names_assigned_in,
    _overlaps_diff,
)


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


def test_names_assigned_in_simple():
    assert _names_assigned_in("x = 1\n") == {"x"}


def test_names_assigned_in_tuple_unpack():
    assert _names_assigned_in("x, y = f()\n") == {"x", "y"}


def test_names_assigned_in_augassign():
    assert _names_assigned_in("x += 1\n") == {"x"}


def test_names_assigned_in_no_assign():
    assert _names_assigned_in("f()\n") == set()


def test_names_assigned_in_syntax_error():
    assert _names_assigned_in("def (\n") == set()


def test_extract_defined_names_basic():
    source = textwrap.dedent(
        """\
        def foo():
            pass

        async def bar():
            pass

        class Baz:
            pass
        """
    )
    assert _extract_defined_names(source) == {"foo", "bar", "Baz"}


def test_extract_defined_names_syntax_error():
    assert _extract_defined_names("def (\n") == set()


def _make_esc_seq(start: int, end: int) -> _SeqInfo:
    """Create a _SeqInfo for escaping-vars tests."""
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source="",
        fingerprint="",
    )


def test_find_escaping_vars_no_assignments():
    # Block has no assignments → skip (branch A), returns empty set.
    source_lines = [
        "def foo():\n",
        "    compute()\n",
        "    transform()\n",
        "    use_result()\n",
    ]
    seq = _make_esc_seq(2, 3)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_nothing_after_block():
    # Block is the last thing in scope → after_lines empty (branch D), returns set().
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_escapes():
    # Block assigns z; z is used after the block → {"z"}.
    # Also covers: blank line (branch B) and lower-indent stop (branch C).
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",  # block ends line 4
        "\n",  # blank → branch B
        "    assert z == 42\n",  # same indent, uses z
        "\n",
        "def bar():\n",  # indent 0 < 4 → branch C (stop)
        "    pass\n",
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == {"z"}


def test_find_escaping_vars_no_escape():
    # Block assigns x/y/z; none referenced after the block → set().
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",
        "    print('done')\n",  # uses 'print', not x/y/z
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_syntax_error_after():
    # After source is invalid Python → SyntaxError branch: continue, returns set().
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",
        "    def bar(x\n",  # unclosed paren at same indent
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_module_level_stops_at_def():
    # Module-level block (indent 0): a non-def/class line is included,
    # then a def line stops the scan (break via re.match).
    source_lines = [
        "x = compute()\n",
        "y = transform(x)\n",
        "z = finalize(y)\n",  # block ends line 3
        "CONSTANT = 42\n",  # module-level non-def → appended (False branch of re.match)
        "def foo(z):\n",  # module-level def → stop
        "    return z\n",
    ]
    seq = _make_esc_seq(1, 3)
    # CONSTANT is in after_lines; not in assigned → set().
    # z inside def foo(z) is not scanned (stopped before that def).
    assert _find_escaping_vars([seq], source_lines) == set()


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


def _collect_functions(source: str):
    tree = cst.parse_module(source)
    lines = source.splitlines(keepends=True)
    collector = _FunctionCollector(lines)
    MetadataWrapper(tree).visit(collector)
    return collector.functions


def test_function_collector_module_level():
    source = "def foo():\n    pass\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert funcs[0].name == "foo"
    assert funcs[0].scope == "<module>"
    assert funcs[0].body_stmt_count == 1
    assert funcs[0].params == []


def test_function_collector_class_level():
    source = "class C:\n    def method(self):\n        pass\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert funcs[0].name == "method"
    assert funcs[0].scope == "C"
    assert funcs[0].body_stmt_count == 1
    assert funcs[0].params == ["self"]


def test_function_collector_skips_nested():
    source = "def outer():\n    def inner():\n        pass\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert funcs[0].name == "outer"
    assert funcs[0].body_stmt_count == 1
    assert funcs[0].params == []


def test_function_collector_collects_body_source():
    source = "def foo():\n    x = 1\n    y = 2\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert "x = 1" in funcs[0].body_source


def test_function_collector_collects_stmt_count():
    source = "def foo():\n    pass\n"
    funcs = _collect_functions(source)
    assert funcs[0].body_stmt_count == 1


def test_function_collector_collects_params():
    source = "def f(x, y):\n    pass\n"
    funcs = _collect_functions(source)
    assert funcs[0].params == ["x", "y"]


def test_function_collector_no_params():
    source = "def f():\n    pass\n"
    funcs = _collect_functions(source)
    assert funcs[0].params == []


def test_sequence_collector_class_scope():
    """_SequenceCollector sets class_scope for sequences inside class methods."""

    source = textwrap.dedent(
        """\
        x = 1
        y = 2
        z = 3

        class MyClass:
            def method(self):
                a = 1
                b = 2
                c = 3
        """
    )
    lines = source.splitlines(keepends=True)
    tree = cst.parse_module(source)
    collector = _SequenceCollector(lines, max_seq_len=8)
    MetadataWrapper(tree).visit(collector)

    module_seqs = [s for s in collector.sequences if s.class_scope is None]
    class_seqs = [s for s in collector.sequences if s.class_scope == "MyClass"]
    assert module_seqs, "expected module-level sequences with class_scope=None"
    assert class_seqs, "expected class-method sequences with class_scope='MyClass'"


def _make_seq_info(start: int, end: int, src: str = "") -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source=src,
        fingerprint="",
    )


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
