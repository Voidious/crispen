from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _FunctionInfo,
    _SeqInfo,
    _would_create_proxy_wrappers,
)
from .test_core_extraction import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)


# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

    def foo():
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_COLLISION_RANGES = [(12, 14)]  # overlaps bar's duplicate block


def test_extraction_name_collision_skipped(monkeypatch, capsys):
    # LLM returns function_name="_helper", which is already defined → skipped.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
                    "helper_source": "def _helper(x, y):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data, x)\n",
                        "    _helper(data, x)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _COLLISION_RANGES,
            source=_COLLISION_SOURCE,
            verbose=True,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None
    assert de.changes_made == []
    err = capsys.readouterr().err
    assert "name collision" in err
    assert "_helper" in err


def test_extraction_name_collision_silent(monkeypatch, capsys):
    # Same collision, verbose=False → no stderr output.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
                    "helper_source": "def _helper(x, y):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data, x)\n",
                        "    _helper(data, x)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _COLLISION_RANGES,
            source=_COLLISION_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None
    assert de.changes_made == []
    err = capsys.readouterr().err
    assert "name collision" not in err


def test_duplicate_extractor_helper_docstrings_false_strips_docstring(
    monkeypatch, capsys
):
    """When helper_docstrings=False, the LLM-returned docstring is stripped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_shared",
                    "placement": "module_level",
                    "helper_source": (
                        "def _shared(data):\n"
                        '    """LLM added a docstring."""\n'
                        "    pass\n"
                    ),
                    "call_site_replacements": [
                        "    _shared(data)\n",
                        "    _shared(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, helper_docstrings=False
        )

    assert de._new_source is not None
    assert '"""LLM added a docstring."""' not in de._new_source


def test_duplicate_extractor_helper_docstrings_true_keeps_docstring(
    monkeypatch, capsys
):
    """When helper_docstrings=True, the LLM-returned docstring is preserved."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_shared",
                    "placement": "module_level",
                    "helper_source": (
                        "def _shared(data):\n"
                        '    """Keep this docstring."""\n'
                        "    pass\n"
                    ),
                    "call_site_replacements": [
                        "    _shared(data)\n",
                        "    _shared(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, helper_docstrings=True
        )

    assert de._new_source is not None
    assert '"""Keep this docstring."""' in de._new_source


_RETURN_BLOCK_SOURCE = textwrap.dedent(
    """\
    def foo():
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        return y

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        return y
    """
)
_RETURN_BLOCK_RANGES = [(10, 12)]  # overlaps bar's duplicate block (x/y/return lines)


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
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def test_b(mock_client):
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_PARAM_DUP_RANGES = [(10, 12)]  # overlaps test_b's duplicate block


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
