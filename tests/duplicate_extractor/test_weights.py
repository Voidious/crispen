import textwrap
from unittest.mock import MagicMock, patch
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    _FunctionCollector,
    _SeqInfo,
    _SequenceCollector,
    _has_def,
    _find_escaping_vars,
    _node_weight,
    _normalize_source,
    _sequence_weight,
    _seq_ends_with_return,
    _replacement_contains_return,
    _replacement_steals_post_block_line,
    DuplicateExtractor,
)
from .test_extraction_verification import _make_extract_response, _make_veto_response
from .test_normalization_indentation import _make_seq_with_source


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


def test_no_source_no_analysis():
    de = DuplicateExtractor([(1, 5)])
    assert de._new_source is None
    assert de.get_rewritten_source() is None


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


def _make_steal_seq(end_line: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=end_line, scope="f", source="", fingerprint=""
    )


def test_replacement_steals_post_block_at_eof():
    # Block is the last line of the file — no post-block line exists.
    source_lines = ["x = 1\n"]
    seq = _make_steal_seq(1)  # next_idx=1 >= len=1 → skip
    assert not _replacement_steals_post_block_line(
        [seq], ["y = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_blank_after():
    # Post-block line is blank but there is a non-blank line further down.
    # The check must scan past the blank to find the real post-block code.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "\n" → scan → next_idx=2 → "y = 2"
    assert _replacement_steals_post_block_line([seq], ["y = 2\n"], source_lines)


def test_replacement_steals_post_block_blank_after_no_match():
    # Blank after block, but replacement doesn't steal the non-blank post-block line.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_all_blank_after():
    # Only blank lines follow the block — no real post-block line to steal.
    source_lines = ["x = 1\n", "\n", "\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_no_match():
    # Replacement last line doesn't match post-block line.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_match():
    # Replacement last line matches post-block line → steal detected.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert _replacement_steals_post_block_line(
        [seq], ["z = helper()\ny = 2\n"], source_lines
    )


_RETURN_BLOCK_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        return y

    def bar():
        x = compute(data)
        y = transform(x)
        return y
    """
)
_RETURN_BLOCK_RANGES = [(7, 9)]  # overlaps bar's body


def _make_return_block_extract_response():
    return _make_extract_response(
        {
            "function_name": "_helper",
            "placement": "module_level",
            "helper_source": (
                "def _helper():\n"
                "    x = compute(data)\n"
                "    y = transform(x)\n"
                "    return y\n"
            ),
            # replacement drops the return — this is the bug being guarded
            "call_site_replacements": [
                "    _helper()\n",
                "    _helper()\n",
            ],
        }
    )


def test_block_ends_with_return_guard_skips(monkeypatch, capsys):
    """Extraction rejected when block ends with return but replacement omits it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
    assert "block ends with return but replacement omits it" in capsys.readouterr().err


def test_block_ends_with_return_guard_skips_silent(monkeypatch):
    """verbose=False: extraction rejected with no stderr output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None


_PARAM_DUP_SOURCE = textwrap.dedent(
    """\
    def test_a(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def test_b(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_PARAM_DUP_RANGES = [(7, 9)]  # overlaps test_b's body


def _make_import_local_extract_response():
    return _make_extract_response(
        {
            "function_name": "_helper",
            "placement": "module_level",
            # helper imports mock_client instead of taking it as a parameter
            "helper_source": (
                "def _helper():\n"
                "    import mock_client\n"
                "    x = compute(data)\n"
                "    y = transform(x)\n"
                "    z = finalize(y)\n"
            ),
            "call_site_replacements": [
                "    _helper()\n",
                "    _helper()\n",
            ],
        }
    )


def test_helper_imports_local_guard_skips(monkeypatch, capsys):
    """Extraction rejected when helper imports a name that is a param in original."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_import_local_extract_response(),
        ]
        de = DuplicateExtractor(
            _PARAM_DUP_RANGES,
            source=_PARAM_DUP_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
    assert "helper imports a name that is a parameter/local" in capsys.readouterr().err


def test_helper_imports_local_guard_skips_silent(monkeypatch):
    """verbose=False: extraction rejected with no stderr output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_import_local_extract_response(),
        ]
        de = DuplicateExtractor(
            _PARAM_DUP_RANGES,
            source=_PARAM_DUP_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
