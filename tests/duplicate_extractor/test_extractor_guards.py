from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _FunctionInfo,
    _SeqInfo,
    _has_param_overwritten_before_read,
    _missing_free_vars,
    _pyflakes_new_undefined_names,
    _would_create_proxy_wrappers,
)
from .test_extractor_core import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)


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


def test_pyflakes_new_undefined_names_returns_empty_when_no_new_issues():
    # Names undefined in both original and candidate → no NEW issues.
    original = "def foo():\n    return bar()\n"
    candidate = "def _h():\n    pass\n\ndef foo():\n    return bar()\n"
    assert _pyflakes_new_undefined_names(original, candidate) == set()


def test_pyflakes_new_undefined_names_detects_introduced_name():
    # candidate introduces a reference to an unassigned name not in original.
    original = "def foo():\n    x = 1\n    return x\n"
    # candidate removes the assignment, leaving x undefined at the call site
    candidate = "def _h():\n    x = 1\n\ndef foo():\n    _h(x)\n    return x\n"
    assert "x" in _pyflakes_new_undefined_names(original, candidate)


def test_missing_free_vars_catches_missing_name():
    # The exact bug pattern: `new_source` is a local variable read in the
    # original block, but the LLM turned it into `transformer.new_source`
    # (an attribute access).  Neither the call site nor the helper body contain
    # a bare `new_source` Name node.
    source = (
        "def run(transformer, file_msgs, filepath):\n"
        "    new_source = get_source()\n"
        "    current_source = new_source\n"
    )
    block_src = "    current_source = new_source\n"
    call_src = "    current_source = _h(transformer, filepath, file_msgs)\n"
    helper_src = (
        "def _h(transformer, filepath, file_msgs):\n"
        "    return transformer.new_source\n"
    )
    assert "new_source" in _missing_free_vars(block_src, [call_src], helper_src, source)


def test_missing_free_vars_no_missing_when_passed_as_arg():
    # Free var is passed as an argument to the helper → not missing.
    source = (
        "def run():\n    new_source = get_source()\n    current_source = new_source\n"
    )
    block_src = "    current_source = new_source\n"
    call_src = "    current_source = _h(new_source)\n"
    helper_src = "def _h(new_source):\n    return new_source\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_ignores_block_locals():
    # `x` is assigned AND read within the block — it is a local, not a free
    # variable.  It should not be flagged even if it's absent from the helper.
    source = "def run():\n    x = 1\n    result = x + 1\n"
    block_src = "    x = 1\n    result = x + 1\n"
    call_src = "    result = _h()\n"
    helper_src = "def _h():\n    x = 1\n    return x + 1\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_ignores_module_level_names():
    # `compute`, `transform`, `finalize` are module-level function names that
    # are never assigned anywhere — the helper can reference them directly.
    source = (
        "def foo():\n"
        "    x = compute(data)\n"
        "    y = transform(x)\n"
        "    z = finalize(y)\n"
    )
    block_src = "    x = compute(data)\n    y = transform(x)\n    z = finalize(y)\n"
    call_src = "    _helper(data)\n"
    helper_src = "def _helper(data):\n    pass\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_syntax_error_in_block_returns_empty():
    assert (
        _missing_free_vars("not valid python!!!", ["x = 1\n"], "def f(): pass\n", "")
        == set()
    )


def test_missing_free_vars_syntax_error_in_replacement_returns_empty():
    source = "def run():\n    a = 1\n"
    assert (
        _missing_free_vars("x = a\n", ["not valid!!!\n"], "def f(): pass\n", source)
        == set()
    )


def test_missing_free_vars_syntax_error_in_source_returns_empty():
    assert (
        _missing_free_vars("x = a\n", ["y = a\n"], "def f(a): pass\n", "not valid!!!")
        == set()
    )


def test_missing_free_vars_empty_block_returns_empty():
    # A block with no reads has no free vars → nothing can be missing.
    source = "def run():\n    x = 1\n"
    block_src = "    x = 1\n"
    call_src = "    _h()\n"
    helper_src = "def _h():\n    x = 1\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_function_parameter_is_caught():
    # A function parameter that's free in the block must appear in the
    # replacement — parameters are local to the function and cannot be
    # accessed by a helper without being passed as an argument.
    source = "def run(verbose):\n    msg = verbose\n"
    block_src = "    msg = verbose\n"
    call_src = "    msg = _h()\n"
    helper_src = "def _h():\n    pass\n"
    assert "verbose" in _missing_free_vars(block_src, [call_src], helper_src, source)


def _make_pyflakes_check_extractor(monkeypatch, verbose=True):
    """Helper: extraction that passes compile() but pyflakes finds a new undefined
    name."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor.extractor._pyflakes_new_undefined_names",
            return_value={"mock_client"},
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(x):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


def test_pyflakes_check_skips_group_verbose(monkeypatch, capsys):
    # Pyflakes finds a new undefined name → group is skipped (verbose path).
    de = _make_pyflakes_check_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert (
        "undefined name(s) introduced by edit: mock_client" in capsys.readouterr().err
    )


def test_pyflakes_check_skips_group_verbose_false(monkeypatch):
    # verbose=False: pyflakes failure is silent.
    de = _make_pyflakes_check_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def _make_missing_free_vars_extractor(monkeypatch, verbose=True):
    """Helper: extraction that passes all earlier guards but _missing_free_vars
    detects a free variable absent from the replacement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor.extractor._missing_free_vars",
            return_value={"new_source"},
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(x):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


def test_missing_free_vars_check_skips_group_verbose(monkeypatch, capsys):
    # _missing_free_vars returns a non-empty set → group is rejected (verbose).
    de = _make_missing_free_vars_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert (
        "free variable(s) from original block missing in replacement: new_source"
        in capsys.readouterr().err
    )


def test_missing_free_vars_check_skips_group_verbose_false(monkeypatch):
    # verbose=False: _missing_free_vars failure is silent.
    de = _make_missing_free_vars_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_verify_fails_skipped(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "helper",
                    "placement": "module_level",
                    "helper_source": "def helper(x:\n    pass\n",  # unclosed paren
                    "call_site_replacements": [
                        "helper(data)\n",
                        "helper(data)\n",
                    ],
                }
            ),
        ]

        de = DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_verify_fails_skipped_verbose_false(monkeypatch):
    # verbose=False covers the False branch of the new if-self.verbose guard.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "helper",
                    "placement": "module_level",
                    "helper_source": "def helper(x:\n    pass\n",  # unclosed paren
                    "call_site_replacements": [
                        "helper(data)\n",
                        "helper(data)\n",
                    ],
                }
            ),
        ]

        de = DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def _make_new_attr_extractor(monkeypatch, verbose=True):
    """Helper: LLM returns a helper that calls a method not in the original source."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "helper",
                    "placement": "module_level",
                    # helper calls .invented_method() — not present in _DUP_SOURCE
                    "helper_source": (
                        "def helper(data):\n" "    data.invented_method()\n"
                    ),
                    "call_site_replacements": [
                        "helper(data)\n",
                        "helper(data)\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


def test_new_attribute_check_skips_group_verbose(monkeypatch, capsys):
    de = _make_new_attr_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "new attribute access" in capsys.readouterr().err


def test_new_attribute_check_skips_group_verbose_false(monkeypatch):
    de = _make_new_attr_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_llm_name_without_underscore_is_prefixed(monkeypatch):
    """LLM returns a name without a leading '_'; extractor prepends one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "helper",  # no underscore
                    "placement": "module_level",
                    "helper_source": "def helper(data):\n    pass\n",
                    "call_site_replacements": [
                        "    helper(data)\n",
                        "    helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, extraction_retries=0, llm_verify_retries=0
        )

    assert de._new_source is not None
    assert "def _helper(" in de._new_source
    assert "def helper(" not in de._new_source
    assert "_helper(data)" in de._new_source


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
