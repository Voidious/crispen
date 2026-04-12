from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _ApiTimeout,
    _SeqInfo,
    _collect_called_names,
    _extract_defined_names,
    _has_def,
    _lift_and_dedup_imports,
    _names_assigned_in,
    _names_in_edit_texts,
    _node_weight,
    _normalize_replacement_indentation,
    _normalize_source,
    _run_with_timeout,
    _scope_end_line,
    _sequence_weight,
)
import libcst as cst
import pytest
from .test_llm_integration import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _make_seq_info,
    _make_verify_response,
)


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


def _make_seq_with_source(source: str) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=1, scope="f", source=source, fingerprint=""
    )


def test_normalize_indentation_already_correct():
    # Replacement already matches the block's indentation — unchanged.
    seq = _make_seq_with_source("    x = compute()\n    y = finalize(x)\n")
    replacement = "    result = helper()\n"
    assert (
        _normalize_replacement_indentation(seq, replacement)
        == "    result = helper()\n"
    )


def test_normalize_indentation_col0_to_indented():
    # Replacement at column 0 is re-indented to match the original block.
    seq = _make_seq_with_source("    x = compute()\n    y = finalize(x)\n")
    replacement = "result = helper()\n"
    assert (
        _normalize_replacement_indentation(seq, replacement)
        == "    result = helper()\n"
    )


def test_normalize_indentation_multiline():
    # Multi-line replacement at column 0 gets uniformly re-indented.
    seq = _make_seq_with_source("        x = a()\n        y = b(x)\n")
    replacement = "x = helper()\nif x is None:\n    x = default()\n"
    expected = (
        "        x = helper()\n        if x is None:\n            x = default()\n"
    )
    assert _normalize_replacement_indentation(seq, replacement) == expected


def test_normalize_indentation_module_level_block():
    # Module-level block (no indent) — replacement is just dedented.
    seq = _make_seq_with_source("x = compute()\ny = finalize(x)\n")
    replacement = "result = helper()\n"
    assert _normalize_replacement_indentation(seq, replacement) == "result = helper()\n"


def test_normalize_indentation_empty_source():
    # Empty source — no indentation can be inferred; replacement returned as-is.
    seq = _make_seq_with_source("")
    replacement = "result = helper()\n"
    assert _normalize_replacement_indentation(seq, replacement) == replacement


def test_names_in_edit_texts_collects_from_all_edits():
    groups = [
        (
            "_helper",
            [
                (1, 3, "def _helper(last_import_line):\n    return last_import_line\n"),
                (5, 6, "result = _helper(x)\n"),
            ],
            "msg",
        )
    ]
    names = _names_in_edit_texts(groups)
    assert "last_import_line" in names
    assert "_helper" in names
    assert "result" in names
    assert "x" in names


def test_names_in_edit_texts_skips_syntax_errors():
    groups = [("_h", [(1, 2, "def (\n")], "msg")]
    # Should not raise — returns whatever names were parseable.
    names = _names_in_edit_texts(groups)
    assert isinstance(names, set)


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


def test_run_with_timeout_fires_on_slow_func():
    import threading

    barrier = threading.Event()
    try:
        with pytest.raises(_ApiTimeout):
            _run_with_timeout(barrier.wait, timeout=0.01)
    finally:
        barrier.set()  # allow the daemon thread to exit cleanly


def test_veto_timeout_skips_group(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            side_effect=_ApiTimeout("veto timed out"),
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)
    assert de._new_source is None
    assert de.changes_made == []


def test_extract_timeout_skips_group(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # First call (veto) returns success; second call (extract) times out.
    side_effects = [(True, "same logic", ""), _ApiTimeout("extract timed out")]

    def _mock_run(func, timeout, *args, **kwargs):
        result = side_effects.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)
    assert de._new_source is None


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


def test_lift_and_dedup_no_changes_needed():
    src = "import os\nfrom typing import Any, Dict\nx = 1\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_exact_from_duplicate():
    src = "from typing import Any\nfrom typing import Any\n"
    assert _lift_and_dedup_imports(src) == "from typing import Any\n"


def test_lift_and_dedup_partial_overlap_adds_new_names():
    # Original F811 trigger: helper adds Any+Dict+Optional, file had Any+Dict
    src = "from typing import Any, Dict\nfrom typing import Any, Dict, Optional\n"
    assert _lift_and_dedup_imports(src) == "from typing import Any, Dict, Optional\n"


def test_lift_and_dedup_second_adds_only_new_names():
    src = "from typing import Any\nfrom typing import Optional\n"
    assert _lift_and_dedup_imports(src) == "from typing import Any, Optional\n"


def test_lift_and_dedup_multiple_modules_independent():
    src = (
        "from typing import Any\n"
        "from os.path import join\n"
        "from typing import Dict\n"
        "from os.path import exists\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == "from typing import Any, Dict\nfrom os.path import join, exists\n"


def test_lift_and_dedup_plain_import_deduped():
    # Unlike the old _dedup_from_imports, plain 'import X' dups are now removed
    src = "import os\nimport os\n"
    assert _lift_and_dedup_imports(src) == "import os\n"


def test_lift_and_dedup_skips_multiline_parens():
    src = "from typing import (\n    Any,\n    Dict,\n)\nfrom typing import Any\n"
    # Paren form not matched; single-line import stands alone — no change
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_skips_wildcard():
    src = "from typing import *\nfrom typing import *\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_skips_commented_import_line():
    # Inline comment prevents matching; both lines are left alone
    src = "from typing import Any  # noqa\nfrom typing import Any\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_skips_indented_imports():
    # Indented imports (TYPE_CHECKING blocks, try/except, etc.) are not touched
    src = "    from typing import Any\n    from typing import Dict\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_empty_names_skipped():
    # Malformed import with no names: left unchanged
    src = "from typing import ,\nfrom typing import ,\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_non_import_lines_preserved():
    src = "from typing import Any\nx = 1\nfrom typing import Dict\ny = 2\n"
    result = _lift_and_dedup_imports(src)
    assert result == "from typing import Any, Dict\nx = 1\ny = 2\n"


def test_lift_and_dedup_lifts_misplaced_existing_module():
    # Helper inserted before second_fn lands after def first_fn → misplaced
    # The import merges into the block and the misplaced copy is removed.
    src = (
        "from typing import Any\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "from typing import Optional\n"  # misplaced — helper preamble
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any, Optional\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )


def test_lift_and_dedup_lifts_misplaced_new_module():
    # Helper introduces a brand-new import mid-file → moved to after block.
    src = (
        "from typing import Any\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "from collections import OrderedDict\n"  # misplaced — new module
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any\n"
        "from collections import OrderedDict\n"  # lifted after last block import
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )


def test_lift_and_dedup_lifts_misplaced_plain_import_new_module():
    # Covers: misplaced plain 'import X' (i >= first_funcdef_idx branch) and
    # the new_plain_modules emission path inside _emit_new_imports.
    src = (
        "from typing import Any\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "import os\n"  # misplaced plain import — new module
        "def _helper():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any\n"
        "import os\n"  # lifted after last block import
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
    )


def test_lift_and_dedup_sorts_new_imports_by_pep8_section():
    # New lifted imports are sorted future→stdlib→third-party→local regardless
    # of the order they were encountered.
    src = (
        "from typing import Any\n"  # block stdlib import
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "import requests\n"  # misplaced third-party
        "from collections import OrderedDict\n"  # misplaced stdlib
        "def _helper():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any\n"
        "from collections import OrderedDict\n"  # stdlib before third-party
        "import requests\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
    )


def test_lift_and_dedup_blank_lines_in_block_dropped():
    # Blank lines between import lines in the block are removed when the block
    # is rebuilt — covers the blank-line-dropping branch in pass 5.
    src = (
        "import os\n"
        "\n"  # blank between block imports → dropped on rebuild
        "from typing import Any\n"
        "from typing import Dict\n"  # duplicate module → merged
        "x = 1\n"
    )
    result = _lift_and_dedup_imports(src)
    # PEP 8 sort: both are stdlib (group 1); from_order precedes plain_order in
    # all_final_imports so stable sort keeps 'from typing' before 'import os'.
    assert result == ("from typing import Any, Dict\n" "import os\n" "x = 1\n")


def test_lift_and_dedup_no_block_imports_inserts_before_first_funcdef():
    # File has no imports at all; helper adds one mid-file → moved to very top.
    src = (
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "from collections import OrderedDict\n"  # misplaced
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from collections import OrderedDict\n"  # inserted before first funcdef
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )


def test_llm_verify_extraction_with_timing_out():
    """_llm_verify_extraction appends result to _timing_out when provided."""
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    client = MagicMock()
    client.messages.create.return_value = _make_verify_response(True, [])
    group = [_make_seq_info(1, 3), _make_seq_info(5, 7)]
    timing: list = []
    is_correct, issues = _llm_verify_extraction(
        client,
        group,
        "def _helper(): pass\n",
        ["    _helper()\n", "    _helper()\n"],
        "a = 1\nb = 2\n",
        _timing_out=timing,
    )
    assert is_correct is True
    assert len(timing) == 1


def test_llm_verify_extraction_without_timing_out():
    """_llm_verify_extraction works correctly when _timing_out is None."""
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    client = MagicMock()
    client.messages.create.return_value = _make_verify_response(True, [])
    group = [_make_seq_info(1, 3), _make_seq_info(5, 7)]
    is_correct, issues = _llm_verify_extraction(
        client,
        group,
        "def _helper(): pass\n",
        ["    _helper()\n", "    _helper()\n"],
        "a = 1\nb = 2\n",
    )
    assert is_correct is True
    assert issues == []
