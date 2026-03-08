from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _ApiTimeout,
    _collect_attribute_names,
    _collect_called_attr_names,
    _collect_called_names,
    _has_call_to,
    _has_mutable_literal_is_check,
    _has_param_overwritten_before_read,
    _verify_extraction,
)
from ..test_duplicate_extractor import _DUP_RANGES, _DUP_SOURCE


def _make_verify_response(is_correct: bool, issues: list) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "verify_extraction"
    block.input = {"is_correct": is_correct, "issues": issues}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_extract_response(data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "extract_helper"
    block.input = data
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_veto_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_verify_extraction_valid():
    helper = "def helper(x):\n    return x + 1\n"
    replacements = ["result = helper(a)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_invalid_helper():
    helper = "def helper(x:\n    pass\n"  # unclosed paren → syntax error after dedent
    replacements = ["result = helper(a)\n"]
    assert _verify_extraction(helper, replacements) is False


def test_verify_extraction_invalid_replacement():
    helper = "def helper(x):\n    return x\n"
    # Dedented replacement still has a syntax error
    replacements = ["result = helper(a\n"]  # unclosed paren
    assert _verify_extraction(helper, replacements) is False


def test_verify_extraction_no_helper_source():
    # Exercises the helper_source is None branch (skips helper compile check).
    assert _verify_extraction(None, ["result = f()\n"]) is True


def test_verify_extraction_fails_on_param_overwrite():
    # Helper where the parameter is immediately overwritten before being read.
    helper = "def setup(mock_obj):\n    mock_obj = object()\n    return mock_obj\n"
    assert _verify_extraction(helper, ["x = setup(y)\n"]) is False


def test_verify_extraction_allows_return_in_replacement():
    # Replacements inside function bodies legally contain 'return'; the dummy-
    # function wrapper must allow this without triggering a false rejection.
    helper = "def helper(x):\n    return x\n"
    replacements = ["    return helper(a)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_multiline_return_replacement():
    # Multi-line replacement ending with a return statement.
    helper = "def helper(source):\n    return helper(source)\n"
    replacements = [
        "    tree = helper(source)\n    if tree is None:\n        return set()\n"
    ]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_continue_in_replacement():
    # 'continue' is valid inside a loop body; the dummy wrapper now includes a
    # for loop so this is not rejected as a SyntaxError.
    helper = "def helper():\n    pass\n"
    replacements = ["    if done:\n        continue\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_break_in_replacement():
    # Same as above but for 'break'.
    helper = "def helper():\n    pass\n"
    replacements = ["    if done:\n        break\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_await_in_replacement():
    # Replacements inside async functions legally contain 'await'; the async
    # dummy-function wrapper must allow this without triggering a false rejection.
    helper = "async def helper(x):\n    return await x\n"
    replacements = ["    result = await helper(coro)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_async_helper():
    # async def helpers are valid Python and must compile successfully.
    helper = "async def helper(client, x):\n    return await client.get(x)\n"
    replacements = ["    val = await helper(client, url)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_rejects_invalid_await_replacement():
    # Replacement with `await` that also has a real syntax error must still fail.
    helper = "async def helper(x):\n    return await x\n"
    replacements = ["    result = await helper(coro\n"]  # unclosed paren
    assert _verify_extraction(helper, replacements) is False


def test_has_mutable_literal_is_check_set_constructor():
    assert _has_mutable_literal_is_check("if x is set(): pass") is True


def test_has_mutable_literal_is_check_list_constructor():
    assert _has_mutable_literal_is_check("if x is list(): pass") is True


def test_has_mutable_literal_is_check_dict_constructor():
    assert _has_mutable_literal_is_check("if x is dict(): pass") is True


def test_has_mutable_literal_is_check_list_literal():
    assert _has_mutable_literal_is_check("if x is []: pass") is True


def test_has_mutable_literal_is_check_dict_literal():
    assert _has_mutable_literal_is_check("if x is {}: pass") is True


def test_has_mutable_literal_is_check_isnot():
    assert _has_mutable_literal_is_check("if x is not set(): pass") is True


def test_has_mutable_literal_is_check_none_is_fine():
    assert _has_mutable_literal_is_check("if x is None: pass") is False


def test_has_mutable_literal_is_check_isinstance_is_fine():
    assert _has_mutable_literal_is_check("if isinstance(x, set): pass") is False


def test_has_mutable_literal_is_check_equality_is_fine():
    # == comparison with set() is valid; only identity (`is`) is wrong
    assert _has_mutable_literal_is_check("if x == set(): pass") is False


def test_has_mutable_literal_is_check_syntax_error():
    assert _has_mutable_literal_is_check("def f(x:") is False


def test_verify_extraction_rejects_mutable_is_in_helper():
    helper = "def h(x):\n    if x is set(): return True\n    return False\n"
    assert _verify_extraction(helper, ["h(a)\n"]) is False


def test_verify_extraction_rejects_mutable_is_in_replacement():
    helper = "def h(x):\n    return x\n"
    assert _verify_extraction(helper, ["if r is set(): pass\n"]) is False


def test_verify_extraction_rejects_indented_mutable_is_in_replacement():
    # Indented replacements (function-body code) are wrapped before checking,
    # so `is set()` is caught even when ast.parse would fail on raw indented text.
    helper = "def h(x):\n    return x\n"
    assert _verify_extraction(helper, ["    if r is set(): pass\n"]) is False


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


def test_no_source_no_analysis():
    de = DuplicateExtractor([(1, 5)])
    assert de._new_source is None
    assert de.get_rewritten_source() is None


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


def test_extraction_retry_on_alg_failure_silent(monkeypatch):
    """First extract has wrong call count -> retry -> second succeeds. verbose=False."""
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
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, extraction_retries=1
        )

    assert de._new_source is not None


def test_llm_verify_timeout_verbose(monkeypatch, capsys):
    """Verify times out (verbose=True) -> extraction is accepted and logged."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    extraction_dict = {
        "function_name": "_helper",
        "placement": "module_level",
        "helper_source": "def _helper(data):\n    pass\n",
        "call_site_replacements": ["    _helper(data)\n", "    _helper(data)\n"],
    }
    side_effects: list = [(True, "same logic", ""), extraction_dict]

    def _mock_run(func, timeout, *args, **kwargs):
        if func is _llm_verify_extraction:
            raise _ApiTimeout("verify timed out")
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, verbose=True)

    assert de._new_source is not None
    err = capsys.readouterr().err
    assert "verify timed out" in err


def test_llm_verify_rejects_then_retries_verbose(monkeypatch, capsys):
    """Verify rejects first attempt; retry extract passes. verbose=True."""
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
            _make_verify_response(False, ["wrong variable name"]),
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
            _DUP_RANGES, source=_DUP_SOURCE, verbose=True, llm_verify_retries=1
        )

    assert de._new_source is not None
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "wrong variable name" in err
    assert "retrying" in err


def test_llm_verify_rejects_then_retries_silent(monkeypatch):
    """Verify rejects first attempt; retry extract passes. verbose=False."""
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
            _make_verify_response(False, ["wrong variable name"]),
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
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, llm_verify_retries=1
        )

    assert de._new_source is not None


def test_llm_verify_exhausted_skips_group(monkeypatch):
    """All verify attempts fail -> group skipped."""
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
            _make_verify_response(False, ["issue"]),
        ]
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, llm_verify_retries=0)

    assert de._new_source is None


def test_llm_verify_timeout_silent(monkeypatch):
    """Verify times out (verbose=False) -> extraction is accepted silently."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    extraction_dict = {
        "function_name": "_helper",
        "placement": "module_level",
        "helper_source": "def _helper(data):\n    pass\n",
        "call_site_replacements": ["    _helper(data)\n", "    _helper(data)\n"],
    }
    side_effects: list = [(True, "same logic", ""), extraction_dict]

    def _mock_run(func, timeout, *args, **kwargs):
        if func is _llm_verify_extraction:
            raise _ApiTimeout("verify timed out")
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, verbose=False)

    assert de._new_source is not None
