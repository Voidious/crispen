import textwrap
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    _FunctionCollector,
    _FunctionInfo,
    _SeqInfo,
    _SequenceCollector,
    _build_function_body_fps,
    _collect_ast_store_names,
    _collect_attribute_names,
    _collect_called_attr_names,
    _collect_called_names,
    _extract_defined_names,
    _filter_maximal_groups,
    _find_duplicate_groups,
    _has_call_to,
    _has_def,
    _has_funcdef,
    _has_internal_overlap,
    _normalize_source,
    _overlaps_diff,
    _seq_source_contains_yield,
)
import libcst as cst


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


def test_collect_attribute_names_basic():
    assert _collect_attribute_names("x.foo()\ny.bar") == {"foo", "bar"}


def test_collect_attribute_names_nested():
    assert "baz" in _collect_attribute_names("a.b.baz()")


def test_collect_attribute_names_syntax_error():
    assert _collect_attribute_names("def f(x:") == set()


def test_collect_attribute_names_no_attrs():
    assert _collect_attribute_names("x = 1 + 2") == set()


def test_collect_called_attr_names_method_call():
    # obj.foo() → "foo" is a called attribute
    assert _collect_called_attr_names("obj.foo()") == {"foo"}


def test_collect_called_attr_names_ignores_plain_access():
    # obj.bar (not called) → not included
    assert "bar" not in _collect_called_attr_names("x = obj.bar")


def test_collect_called_attr_names_ignores_type_annotation():
    # ast.AST used as a type annotation is NOT a method call → not flagged
    assert "AST" not in _collect_called_attr_names(
        "def f(x) -> Optional[ast.AST]: pass"
    )


def test_collect_called_attr_names_syntax_error():
    assert _collect_called_attr_names("def f(x:") == set()


def test_collect_called_attr_names_no_calls():
    assert _collect_called_attr_names("x = 1 + 2") == set()


def test_has_call_to_direct_call():
    assert _has_call_to("foo", "foo()\n") is True


def test_has_call_to_attribute_call():
    assert _has_call_to("foo", "obj.foo()\n") is True


def test_has_call_to_missing():
    assert _has_call_to("foo", "bar()\n") is False


def test_has_call_to_syntax_error():
    assert _has_call_to("foo", "def f(x:") is False


def test_has_funcdef_present():
    assert _has_funcdef("_helper", "def _helper(x):\n    pass\n") is True


def test_has_funcdef_async():
    assert _has_funcdef("_helper", "async def _helper(x):\n    pass\n") is True


def test_has_funcdef_missing():
    assert _has_funcdef("_helper", "def other(x):\n    pass\n") is False


def test_has_funcdef_syntax_error():
    assert _has_funcdef("_helper", "def f(x:") is False


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


def test_collect_called_names_direct():
    names = _collect_called_names("foo()\n")
    assert "foo" in names


def test_collect_called_names_method():
    names = _collect_called_names("obj.bar()\n")
    assert "bar" in names


def test_collect_called_names_empty():
    names = _collect_called_names("x = 1\n")
    assert names == set()


def test_collect_called_names_syntax_error():
    names = _collect_called_names("def f(: pass")
    assert names == set()


def test_collect_called_names_other_callable():
    # func is a subscript (neither Name nor Attribute): funcs[0]()
    # Covers the elif-False branch in _collect_called_names.
    names = _collect_called_names("funcs[0]()\n")
    assert "funcs" not in names  # subscript call adds nothing


def _make_func_info(name: str, body_source: str = "    pass\n") -> _FunctionInfo:
    return _FunctionInfo(
        name=name,
        source=f"def {name}():\n{body_source}",
        scope="<module>",
        body_source=body_source,
        body_stmt_count=1,
        params=[],
    )


def test_build_fps_includes_called():
    body = "    x = 1\n    y = 2\n    z = 3\n"
    func = _make_func_info("foo", body)
    fps = _build_function_body_fps([func], {"foo"})
    fp = _normalize_source(body)
    assert fp in fps
    assert fps[fp].name == "foo"


def test_build_fps_excludes_uncalled():
    func = _make_func_info("bar")
    fps = _build_function_body_fps([func], {"foo"})
    assert fps == {}


def test_build_fps_empty_functions():
    fps = _build_function_body_fps([], {"foo"})
    assert fps == {}


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


def test_collect_ast_store_names_simple_name():
    import ast

    node = ast.parse("x = 1").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert names == ["x"]


def test_collect_ast_store_names_tuple():
    import ast

    node = ast.parse("a, b = 1, 2").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert set(names) == {"a", "b"}


def test_collect_ast_store_names_nested_tuple():
    import ast

    node = ast.parse("(a, (b, c)) = x").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert set(names) == {"a", "b", "c"}


def test_collect_ast_store_names_non_name_non_tuple_noop():
    # ast.Attribute target (e.g. self.x) → nothing collected.
    import ast

    node = ast.parse("self.x = 1").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert names == []


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
