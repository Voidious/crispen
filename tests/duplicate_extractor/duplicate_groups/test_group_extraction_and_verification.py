import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    _SeqInfo,
    _filter_maximal_groups,
    _find_duplicate_groups,
    _overlaps_diff,
    _verify_extraction,
    DuplicateExtractor,
)
from ..test_extraction_verification import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from ..test_weights import _make_seq


def test_overlaps_diff_yes():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(8, 12)]) is True


def test_overlaps_diff_no():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(11, 20)]) is False


def test_overlaps_diff_exact_boundary():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(10, 15)]) is True


def test_find_duplicate_groups_empty():
    assert _find_duplicate_groups([], [(1, 5)]) == []


def test_find_duplicate_groups_singleton():
    seq = _make_seq(1, 3)
    seq.fingerprint = "fp1"
    seqs = [seq]
    # Only one seq with this fingerprint — not a duplicate
    assert _find_duplicate_groups(seqs, [(1, 3)]) == []


def test_find_duplicate_groups_no_diff_overlap():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    # Neither overlaps diff range (20, 30)
    assert _find_duplicate_groups([s1, s2], [(20, 30)]) == []


def test_find_duplicate_groups_valid():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    groups = _find_duplicate_groups([s1, s2], [(1, 3)])
    assert len(groups) == 1
    assert set(id(s) for s in groups[0]) == {id(s1), id(s2)}


def test_find_duplicate_groups_caps_at_max_groups():
    sequences = []
    for i in range(6):
        fp = f"fp{i}"
        # Place each group in a disjoint band of 20 lines so _filter_maximal_groups
        # keeps all 6 (none overlap), and the max_groups=3 cap is what limits output.
        sequences.append(_SeqInfo([], i * 20 + 1, i * 20 + 3, "<module>", "", fp))
        sequences.append(_SeqInfo([], i * 20 + 10, i * 20 + 12, "<module>", "", fp))
    # Diff range covers all sequences so the diff-overlap filter passes for all.
    groups = _find_duplicate_groups(sequences, [(1, 130)], max_groups=3)
    assert len(groups) == 3


def test_filter_maximal_groups_empty():
    assert _filter_maximal_groups([]) == []


def test_filter_maximal_groups_single_group():
    s1 = _SeqInfo([], 1, 10, "<module>", "", "fp1")
    s2 = _SeqInfo([], 20, 29, "<module>", "", "fp1")
    group = [s1, s2]
    result = _filter_maximal_groups([group])
    assert result == [group]


def test_filter_maximal_groups_removes_subsumed():
    # Large group spans lines 1-10; small group spans 1-5 (subset).
    # Only the large group should be kept.
    s_large_a = _SeqInfo([], 1, 10, "<module>", "", "fp_large")
    s_large_b = _SeqInfo([], 20, 29, "<module>", "", "fp_large")
    large_group = [s_large_a, s_large_b]

    s_small_a = _SeqInfo([], 1, 5, "<module>", "", "fp_small")
    s_small_b = _SeqInfo([], 20, 24, "<module>", "", "fp_small")
    small_group = [s_small_a, s_small_b]

    result = _filter_maximal_groups([small_group, large_group])
    assert len(result) == 1
    assert result[0] is large_group


def test_filter_maximal_groups_keeps_non_overlapping():
    # Two groups with completely disjoint line ranges — both should be kept.
    s1a = _SeqInfo([], 1, 5, "<module>", "", "fp1")
    s1b = _SeqInfo([], 30, 34, "<module>", "", "fp1")
    group1 = [s1a, s1b]

    s2a = _SeqInfo([], 10, 14, "<module>", "", "fp2")
    s2b = _SeqInfo([], 40, 44, "<module>", "", "fp2")
    group2 = [s2a, s2b]

    result = _filter_maximal_groups([group1, group2])
    assert len(result) == 2


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


def test_successful_extraction_module_level(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        import os

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

        de = DuplicateExtractor([(9, 11)], source=source)

    assert de._new_source is not None
    assert "_helper" in de._new_source
    assert len(de.changes_made) == 1
    assert "'_helper'" in de.changes_made[0]
    assert de.get_rewritten_source() == de._new_source


def test_staticmethod_placement(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class MyClass:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "    @staticmethod\n    def _helper(data):\n        pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:MyClass",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]

        de = DuplicateExtractor([(8, 10)], source=source)

    assert de._new_source is not None


def test_cross_class_duplicates_use_module_level_placement(monkeypatch):
    """Duplicates in different classes must be extracted as module-level functions."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source)

    assert de._new_source is not None
    # The extraction call prompt should tell the LLM to use module_level
    extract_prompt = mock_client.messages.create.call_args_list[1][1]["messages"][0][
        "content"
    ]
    assert "module_level" in extract_prompt
    assert "staticmethod" not in extract_prompt


def test_cross_class_staticmethod_placement_rejected(monkeypatch):
    """LLM returning staticmethod placement for a cross-class group is rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # First extraction attempt: LLM ignores prompt and returns staticmethod
        # placement for a cross-class group → rejected; second attempt: correct.
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": (
                        "    @staticmethod\n    def _helper(data):\n        pass\n"
                    ),
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source)

    assert de._new_source is not None
    # Three LLM calls: veto + two extraction attempts
    assert mock_client.messages.create.call_count == 4


def test_cross_class_staticmethod_placement_rejected_non_verbose(monkeypatch):
    """Defensive cross-class check works when verbose=False (no print side-effect)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": (
                        "    @staticmethod\n    def _helper(data):\n        pass\n"
                    ),
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source, verbose=False)

    assert de._new_source is not None


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
