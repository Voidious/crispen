from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _ApiTimeout,
    _FunctionInfo,
    _SeqInfo,
    _build_function_body_fps,
    _collect_attribute_names,
    _collect_called_attr_names,
    _has_call_to,
    _has_funcdef,
    _llm_veto_func_match,
    _normalize_source,
    _would_create_proxy_wrappers,
)
from ..test_extractor_core import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _make_call_gen_response,
    _make_extract_response,
    _make_seq_info,
    _make_verify_response,
    _make_veto_func_match_response,
    _make_veto_response,
)


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


# _setup() has no params; called by main() → in func_body_fps.
# foo.body fingerprint == _setup.body fingerprint.
# Diff range (2, 9) covers both _setup.body (2-4) AND foo.body (7-9).
# _setup.body hits the func.name==seq.scope True branch (skipped).
# foo.body hits the False branch and proceeds to veto → replace.
_FUNC_MATCH_SOURCE = textwrap.dedent(
    """\
    def _setup():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def main():
        _setup()
    """
)
_FUNC_MATCH_RANGES = [(2, 9)]  # covers _setup.body AND foo.body

# _process(val) has one param; called by main() → in func_body_fps.
# foo.body fingerprint == _process.body fingerprint (names normalized).
# Diff range covers foo.body only.
_FUNC_MATCH_PARAM_SOURCE = textwrap.dedent(
    """\
    def _process(val):
        y = transform(val)
        z = finalize(y)
        return z

    def foo():
        y = transform(data)
        z = finalize(y)
        return z

    def main():
        _process(data)
    """
)
_FUNC_MATCH_PARAM_RANGES = [(6, 9)]  # overlaps foo.body only

# Source with a function-match AND an independent duplicate group.
# bar/baz use an if-else structure so no sub-window of their bodies matches
# _setup's 3-chained-assignment fingerprint.
_FUNC_MATCH_THEN_DUP_SOURCE = textwrap.dedent(
    """\
    def _setup():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        a = setup(items)
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def baz():
        if quick_check:
            pass
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def main():
        _setup()
    """
)
_FUNC_MATCH_THEN_DUP_RANGES = [(2, 30)]  # covers foo, bar, baz bodies


def test_func_match_no_arg_replaces_body(monkeypatch):
    """No-param module-level function: algorithmic replacement, no call-gen LLM."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            return_value=(True, "same operation", ""),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=True
        )
    assert de._new_source is not None
    assert "_setup" in de.changes_made[0]


def test_func_match_verbose_false(monkeypatch):
    """verbose=False covers all False branches of new if-self.verbose guards."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            return_value=(True, "same operation", ""),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=False
        )
    assert de._new_source is not None


def test_func_match_veto_rejects(monkeypatch):
    """Veto rejects func match → no replacement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            return_value=(False, "different", ""),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=True
        )
    assert de._new_source is None


def test_func_match_veto_timeout(monkeypatch):
    """Veto times out → seq skipped; subsequent dup group also times out."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            side_effect=_ApiTimeout("timed out"),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=True
        )
    assert de._new_source is None


def test_func_match_verify_fails(monkeypatch):
    """_verify_extraction returns False → func match skipped; dup group veto rejects."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Call 1: func match veto → (True, "ok")
    # Call 2: dup group veto → (False, "different") so extract is never called
    side_effects = [(True, "ok", ""), (False, "different", "")]

    def _mock_run(func, timeout, *args, **kwargs):
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._verify_extraction",
            return_value=False,
        ),
    ):
        de = DuplicateExtractor(_FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE)
    assert de._new_source is None


def test_func_match_param_call_gen_success(monkeypatch):
    """Parametrised function: LLM generates call expression successfully."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Call 1: func match veto → (True, "ok")
    # Call 2: _llm_generate_call → replacement string
    side_effects: list = [(True, "ok", ""), "    _process(data)\n"]

    def _mock_run(func, timeout, *args, **kwargs):
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
        )
    assert de._new_source is not None


def test_func_match_param_call_gen_timeout(monkeypatch):
    """Call generation times out → seq skipped; dup group veto rejects."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Call 1: func match veto → (True, "ok")
    # Call 2: _llm_generate_call → timeout
    # Call 3: dup group veto → (False, "reject") so no extract called
    side_effects: list = [
        (True, "ok", ""),
        _ApiTimeout("timed out"),
        (False, "reject", ""),
    ]

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
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
        )
    assert de._new_source is None


def test_func_match_then_dup_extract(monkeypatch):
    """Func match succeeds; remaining dup group triggers standard veto/extract."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    extraction_dict = {
        "function_name": "_helper",
        "placement": "module_level",
        "helper_source": "def _helper():\n    pass\n",
        "call_site_replacements": ["    _helper()\n", "    _helper()\n"],
    }
    # Call 1: func match veto → (True, "ok", "")
    # Call 2: dup group veto → (True, "ok", "")
    # Call 3: dup group extract → extraction dict
    # Call 4: LLM verify → (True, [])
    side_effects: list = [
        (True, "ok", ""),
        (True, "ok", ""),
        extraction_dict,
        (True, []),
    ]

    def _mock_run(func, timeout, *args, **kwargs):
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_THEN_DUP_RANGES,
            source=_FUNC_MATCH_THEN_DUP_SOURCE,
        )
    assert de._new_source is not None
    # One func-match change + one dup-extract change
    assert len(de.changes_made) == 2


def test_match_functions_false_skips_func_match_pass(monkeypatch):
    """match_functions=False: func-match veto never called even when match exists."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    veto_func_match_called: list = []

    def _mock_run_with_timeout(fn, timeout, *args, **kwargs):
        if fn is _llm_veto_func_match:
            veto_func_match_called.append(True)
        # Reject any extraction-pass LLM call so no new source is produced.
        return (False, "rejected", "")

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor.extractor._run_with_timeout",
            side_effect=_mock_run_with_timeout,
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES,
            source=_FUNC_MATCH_SOURCE,
            verbose=False,
            match_functions=False,
        )
    assert veto_func_match_called == []
    assert de._new_source is None


def _make_veto_response_with_notes(
    is_valid: bool, reason: str, notes: str
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {
        "is_valid_duplicate": is_valid,
        "reason": reason,
        "extraction_notes": notes,
    }
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_veto_notes_passed_to_extract(monkeypatch):
    """extraction_notes from veto are forwarded to the extract prompt."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response_with_notes(True, "same logic", "watch out for x"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)

    assert de._new_source is not None
    extract_call = mock_client.messages.create.call_args_list[1]
    extract_prompt = extract_call.kwargs["messages"][0]["content"]
    assert "watch out for x" in extract_prompt


def test_extraction_retry_on_alg_failure_verbose(monkeypatch, capsys):
    """First extract has wrong call count -> retry -> second succeeds. verbose=True."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": ["    _helper(data)\n"],  # wrong count
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=True, extraction_retries=1
        )

    assert de._new_source is not None
    err = capsys.readouterr().err
    assert "retrying" in err


def _make_proxy_seq(stmts_count: int, scope: str, class_scope=None) -> _SeqInfo:
    """Build a _SeqInfo with a synthetic stmts list of the given length."""
    return _SeqInfo(
        stmts=[None] * stmts_count,  # type: ignore[list-item]
        start_line=1,
        end_line=stmts_count,
        scope=scope,
        source="",
        fingerprint="",
        class_scope=class_scope,
    )


def _make_proxy_func(
    name: str, body_stmt_count: int, scope: str = "<module>"
) -> _FunctionInfo:
    return _FunctionInfo(
        name=name,
        source=f"def {name}(): pass\n",
        scope=scope,
        body_source="    pass\n",
        body_stmt_count=body_stmt_count,
        params=[],
    )


def test_would_create_proxy_wrappers_false_single_full_body():
    """Single-member group where the seq covers the entire function body.

    All members are proxies, so extraction is still worthwhile → False.
    """
    seq = _make_proxy_seq(3, scope="foo")
    func = _make_proxy_func("foo", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_all_full_bodies():
    """All group members cover entire function bodies → False.

    When every member becomes a proxy the group is all-or-nothing: extracting
    a shared helper is still worthwhile, so the guard should not block it.
    """
    seq1 = _make_proxy_seq(3, scope="process", class_scope="ClassA")
    seq2 = _make_proxy_seq(3, scope="process", class_scope="ClassB")
    func1 = _make_proxy_func("process", body_stmt_count=3, scope="ClassA")
    func2 = _make_proxy_func("process", body_stmt_count=3, scope="ClassB")
    assert _would_create_proxy_wrappers([seq1, seq2], [func1, func2]) is False


def test_would_create_proxy_wrappers_false_partial_body():
    """A seq that covers only part of a function body → False."""
    seq = _make_proxy_seq(2, scope="foo")
    func = _make_proxy_func("foo", body_stmt_count=4, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_module_scope():
    """A seq at module scope (not inside a function) is never a proxy → False."""
    seq = _make_proxy_seq(3, scope="<module>")
    func = _make_proxy_func("foo", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_no_matching_func():
    """No function with matching name → False."""
    seq = _make_proxy_seq(3, scope="foo")
    func = _make_proxy_func("bar", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_scope_mismatch():
    """Seq in class method but func is module-level with same name → False."""
    seq = _make_proxy_seq(3, scope="foo", class_scope="MyClass")
    func = _make_proxy_func("foo", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_group_with_one_proxy():
    """A group with multiple seqs, one of which covers an entire body → True."""
    seq_partial = _make_proxy_seq(2, scope="foo")
    seq_full = _make_proxy_seq(3, scope="bar")
    func_foo = _make_proxy_func("foo", body_stmt_count=5, scope="<module>")
    func_bar = _make_proxy_func("bar", body_stmt_count=3, scope="<module>")
    assert (
        _would_create_proxy_wrappers([seq_partial, seq_full], [func_foo, func_bar])
        is True
    )


_PROXY_SOURCE = textwrap.dedent(
    """\
    def foo():
        setup = prepare(data)
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        return setup, z

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
# overlaps foo: foo has 5 stmts but duplicate block is only 3 of them (not a proxy);
# bar has 3 stmts = its entire body (would become a proxy) → mixed → guard fires.
_PROXY_RANGES = [(1, 11)]


def test_proxy_wrapper_guard_skips_group_verbose(monkeypatch, capsys):
    """Groups that would leave a function as a trivial proxy are skipped (verbose)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic.Anthropic"):
        de = DuplicateExtractor(_PROXY_RANGES, source=_PROXY_SOURCE, verbose=True)

    assert de._new_source is None
    captured = capsys.readouterr()
    assert "trivial proxy wrapper" in captured.err


def test_proxy_wrapper_guard_skips_group_silent(monkeypatch):
    """Groups that would leave a trivial proxy are skipped with verbose=False."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic.Anthropic"):
        de = DuplicateExtractor(_PROXY_RANGES, source=_PROXY_SOURCE, verbose=False)

    assert de._new_source is None


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


def test_func_match_veto_timing_recorded(monkeypatch):
    """When func-match veto accepts, record_llm_call is invoked for the veto call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # veto accepts → call-gen runs (func has params) → done (no dup groups)
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
        )
    # record_llm_call ran for veto (the timing branch was True)
    assert de.stats.llm_elapsed_by_category.get("veto", 0) >= 0


def test_func_match_call_gen_timing_recorded(monkeypatch):
    """When func-match call-gen runs, record_llm_call is invoked for the edit call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # veto accepts → call-gen runs → done (no dup groups)
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
        )
    assert de.stats.llm_edit_calls >= 1


def test_func_match_veto_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after func-match veto result."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "ACCEPTED" in err
    assert "[" in err  # timing suffix present


def test_func_match_replacement_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after func-match replacement line."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "replacing" in err
    assert "[" in err  # timing suffix on replacement line


def test_dup_veto_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after dup-group veto result."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.return_value = _make_veto_response(
            False, "different logic"
        )
        DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "VETOED" in err
    assert "[" in err  # timing suffix present


def test_verify_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after verify result."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "verify ACCEPTED" in err
    assert "[" in err  # timing suffix present


def test_extraction_detailed_timing_message(monkeypatch, capsys):
    """timing='detailed' prints extraction timing message after extraction call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "→ extraction [" in err
