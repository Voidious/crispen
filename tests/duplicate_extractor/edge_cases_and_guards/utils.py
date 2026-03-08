from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import DuplicateExtractor, _SeqInfo
from ..test_extraction_validation import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from ...test_duplicate_extractor import _DUP_RANGES, _DUP_SOURCE
import textwrap


def _make_steal_seq(end_line: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=end_line, scope="f", source="", fingerprint=""
    )


def _make_invalid_assembled_extractor(monkeypatch, verbose=True):
    """Helper: DuplicateExtractor where _apply_edits returns invalid Python."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._apply_edits",
            return_value="def f(:\n    pass\n",  # invalid Python
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


def _make_pyflakes_check_extractor(monkeypatch, verbose=True):
    """Helper: extraction that passes compile() but pyflakes finds a new undefined
    name."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._pyflakes_new_undefined_names",
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


def _make_missing_free_vars_extractor(monkeypatch, verbose=True):
    """Helper: extraction that passes all earlier guards but _missing_free_vars
    detects a free variable absent from the replacement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._missing_free_vars",
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


_POST_STEAL_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        return z

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        logger.info("done")
    """
)
_POST_STEAL_RANGES = [(8, 10)]  # overlaps bar's 3-statement block


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


def _make_no_call_extractor(monkeypatch, verbose=True):
    """Helper: LLM returns call replacements that don't call the helper function."""
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
                    "helper_source": "def _helper(data):\n    pass\n",
                    # Call replacements don't reference _helper at all.
                    "call_site_replacements": [
                        "    pass\n",
                        "    pass\n",
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


def _make_uncalled_in_combined_extractor(monkeypatch, verbose=True):
    """Simulate: per-group call check passes, but combined output lacks the call.

    Achieved by patching _has_call_to: returns True for the per-group check
    (first call) and False for the final combined check (second call).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._has_call_to",
            side_effect=[True, False],
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
                    "helper_source": "def _helper(data):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        return DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, verbose=verbose)


# Source with two structurally distinct duplicate pairs so _find_duplicate_groups
# returns two separate groups.  The groups differ in argument count so that
# _ASTNormalizer produces different fingerprints for each group:
#   group 1 (foo/bar): 3-stmt bodies using 2-argument calls → fingerprint A
#   group 2 (baz/qux): 3-stmt bodies using 3-argument calls → fingerprint B
_TWO_PAIR_SOURCE = textwrap.dedent(
    """\
    import os

    def foo():
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def bar():
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def baz():
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)

    def qux():
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)
    """
)
_TWO_PAIR_RANGES = [(4, 21)]  # overlaps all duplicate sequences


def _make_two_group_drop_extractor(monkeypatch, verbose=True):
    """Two extraction groups; the combined check drops one, exercising line 1533.

    _has_call_to is patched with side_effect=[True, True, True, False]:
    - calls 1-2: per-group checks for each group → both pass
    - call 3: combined check for first group → kept
    - call 4: combined check for second group → dropped
    After the drop, extraction_groups still has one entry, so the inner
    ``for _, g_edits, _ in extraction_groups`` loop runs once (line 1533).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._has_call_to",
            side_effect=[True, True, True, False],
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # Six LLM calls: veto+extract+verify for each of the two groups.
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper1",
                    "placement": "module_level",
                    "helper_source": "def _helper1():\n    pass\n",
                    "call_site_replacements": [
                        "    _helper1()\n",
                        "    _helper1()\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper2",
                    "placement": "module_level",
                    "helper_source": "def _helper2():\n    pass\n",
                    "call_site_replacements": [
                        "    _helper2()\n",
                        "    _helper2()\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        return DuplicateExtractor(
            _TWO_PAIR_RANGES, source=_TWO_PAIR_SOURCE, verbose=verbose
        )


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
