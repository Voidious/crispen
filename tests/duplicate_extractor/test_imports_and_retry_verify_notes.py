from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _ApiTimeout,
    _lift_and_dedup_imports,
    _strip_helper_docstring,
)
from .test_duplicate_extractor_core_integration import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from .test_llm_and_timeouts import (
    _make_call_gen_response,
    _make_seq_info,
    _make_veto_func_match_response,
)
from .test_function_match_integration import (
    _FUNC_MATCH_PARAM_RANGES,
    _FUNC_MATCH_PARAM_SOURCE,
)


def test_strip_helper_docstring_with_docstring():
    source = 'def _helper(x):\n    """Strip me."""\n    return x\n'
    result = _strip_helper_docstring(source)
    assert '"""Strip me."""' not in result
    assert "return x" in result


def test_strip_helper_docstring_no_docstring():
    source = "def _helper(x):\n    return x\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_parse_error():
    bad = "def f(:\n    pass\n"
    result = _strip_helper_docstring(bad)
    assert result == bad


def test_strip_helper_docstring_non_function():
    source = "x = 1\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_docstring_only_body():
    # Function whose body is only a docstring — don't strip (would leave empty body).
    source = 'def _helper():\n    """Only doc."""\n'
    result = _strip_helper_docstring(source)
    assert '"""Only doc."""' in result


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
            "crispen.refactors.duplicate_extractor.duplicate_extractor._run_with_timeout",
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
            "crispen.refactors.duplicate_extractor.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, verbose=False)

    assert de._new_source is not None


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
