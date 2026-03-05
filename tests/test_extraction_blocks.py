import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from tests.test_response_helpers import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)

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


def test_two_groups_one_dropped_combined_check(monkeypatch, capsys):
    """One of two groups is dropped by the combined call check; the other is kept."""
    de = _make_two_group_drop_extractor(monkeypatch, verbose=True)
    assert de._new_source is not None
    assert "DROPPED" in capsys.readouterr().err


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
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def baz():
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def main():
        _setup()
    """
)
_FUNC_MATCH_THEN_DUP_RANGES = [(2, 23)]  # covers foo, bar, baz bodies

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
