import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    _ApiTimeout,
    _llm_veto_func_match,
    DuplicateExtractor,
)
from ..test_extraction_verification import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from ..test_llm_api_behavior import _DUP_RANGES, _DUP_SOURCE
from .test_group_extraction_and_verification import (
    _FUNC_MATCH_PARAM_RANGES,
    _FUNC_MATCH_PARAM_SOURCE,
    _FUNC_MATCH_RANGES,
    _FUNC_MATCH_SOURCE,
    _FUNC_MATCH_THEN_DUP_RANGES,
    _FUNC_MATCH_THEN_DUP_SOURCE,
)


def test_func_match_no_arg_replaces_body(monkeypatch):
    """No-param module-level function: algorithmic replacement, no call-gen LLM."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
        patch(
            "crispen.refactors.duplicate_extractor._verify_extraction",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor._run_with_timeout",
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


# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_COLLISION_RANGES = [(9, 11)]  # overlaps bar's body


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


def test_duplicate_extractor_custom_model_used(monkeypatch):
    """Custom model string is passed to the Anthropic API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.return_value = _make_veto_response(False, "no")
        DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, model="claude-opus-4-6")
    # Verify the custom model was passed
    call_kwargs = mock_client.messages.create.call_args_list[0][1]
    assert call_kwargs["model"] == "claude-opus-4-6"


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
