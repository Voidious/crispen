from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _helper_imports_local_name,
    _replacement_steals_post_block_line,
)
from ..test_extraction_validation import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from .utils import (
    _POST_STEAL_RANGES,
    _POST_STEAL_SOURCE,
    _make_new_attr_extractor,
    _make_no_call_extractor,
    _make_steal_seq,
    _make_two_group_drop_extractor,
    _make_uncalled_in_combined_extractor,
)
import textwrap


def test_replacement_steals_post_block_skipped(monkeypatch):
    """Replacement whose last line matches the post-block line is rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": (
                        "def _do_work(data):\n"
                        "    x = compute(data)\n"
                        "    y = transform(x)\n"
                        "    z = finalize(y)\n"
                    ),
                    "call_site_replacements": [
                        "    _do_work(data)\n    return z\n",  # steals "return z"
                        "    _do_work(data)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _POST_STEAL_RANGES,
            source=_POST_STEAL_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_replacement_steals_post_block_skipped_verbose_false(monkeypatch):
    """verbose=False covers the False branch of the verbose guard."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": (
                        "def _do_work(data):\n"
                        "    x = compute(data)\n"
                        "    y = transform(x)\n"
                        "    z = finalize(y)\n"
                    ),
                    "call_site_replacements": [
                        "    _do_work(data)\n    return z\n",  # steals "return z"
                        "    _do_work(data)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _POST_STEAL_RANGES,
            source=_POST_STEAL_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_new_attribute_check_skips_group_verbose(monkeypatch, capsys):
    de = _make_new_attr_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "new attribute access" in capsys.readouterr().err


def test_new_attribute_check_skips_group_verbose_false(monkeypatch):
    de = _make_new_attr_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_no_call_check_skips_group_verbose(monkeypatch, capsys):
    de = _make_no_call_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "not called in candidate output" in capsys.readouterr().err


def test_no_call_check_skips_group_verbose_false(monkeypatch):
    de = _make_no_call_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_uncalled_in_combined_drops_group_verbose(monkeypatch, capsys):
    de = _make_uncalled_in_combined_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "DROPPED" in capsys.readouterr().err


def test_uncalled_in_combined_drops_group_verbose_false(monkeypatch):
    de = _make_uncalled_in_combined_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_two_groups_one_dropped_combined_check(monkeypatch, capsys):
    """One of two groups is dropped by the combined call check; the other is kept."""
    de = _make_two_group_drop_extractor(monkeypatch, verbose=True)
    assert de._new_source is not None
    assert "DROPPED" in capsys.readouterr().err


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


def test_replacement_steals_post_block_at_eof():
    # Block is the last line of the file — no post-block line exists.
    source_lines = ["x = 1\n"]
    seq = _make_steal_seq(1)  # next_idx=1 >= len=1 → skip
    assert not _replacement_steals_post_block_line(
        [seq], ["y = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_blank_after():
    # Post-block line is blank but there is a non-blank line further down.
    # The check must scan past the blank to find the real post-block code.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "\n" → scan → next_idx=2 → "y = 2"
    assert _replacement_steals_post_block_line([seq], ["y = 2\n"], source_lines)


def test_replacement_steals_post_block_blank_after_no_match():
    # Blank after block, but replacement doesn't steal the non-blank post-block line.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_all_blank_after():
    # Only blank lines follow the block — no real post-block line to steal.
    source_lines = ["x = 1\n", "\n", "\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_no_match():
    # Replacement last line doesn't match post-block line.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_match():
    # Replacement last line matches post-block line → steal detected.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert _replacement_steals_post_block_line(
        [seq], ["z = helper()\ny = 2\n"], source_lines
    )


def test_helper_imports_local_name_true():
    helper = "def _h():\n    import mock_client\n    mock_client.run()\n"
    original = "def test(mock_client):\n    mock_client.run()\n"
    assert _helper_imports_local_name(helper, original) is True


def test_helper_imports_local_name_already_imported_in_original():
    # mock_client is already a top-level import → not a local-only name.
    helper = "def _h():\n    import mock_client\n    mock_client.run()\n"
    original = "import mock_client\ndef test(x):\n    mock_client.run()\n"
    assert _helper_imports_local_name(helper, original) is False


def test_helper_imports_local_name_no_imports_in_helper():
    helper = "def _h():\n    pass\n"
    original = "def test(mock_client):\n    pass\n"
    assert _helper_imports_local_name(helper, original) is False


def test_helper_imports_local_name_syntax_error_helper():
    assert _helper_imports_local_name("def (:\n", "def test(x):\n    pass\n") is False


def test_helper_imports_local_name_syntax_error_original():
    assert _helper_imports_local_name("def _h():\n    import x\n", "(:\n") is False


def test_helper_imports_local_name_from_import_in_helper():
    # "from X import Y" in helper: the tracked name is "Y", not "X".
    # If "Y" is a param in the original, it is flagged.
    helper = "def _h():\n    from pkg import mock_client\n    mock_client.run()\n"
    original = "def test(mock_client):\n    mock_client.run()\n"
    assert _helper_imports_local_name(helper, original) is True


def test_helper_imports_local_name_from_import_in_original():
    # Top-level "from pkg import something" in the original covers the branch
    # in the orig_top_imports loop and prevents false-positive flagging.
    helper = "def _h():\n    import something\n    something.run()\n"
    original = "from pkg import something\ndef test(x):\n    something.run()\n"
    assert _helper_imports_local_name(helper, original) is False


def test_helper_imports_local_name_vararg():
    # Function with *args: vararg name tracked as potential local.
    helper = "def _h():\n    import args\n"
    original = "def test(*args):\n    pass\n"
    assert _helper_imports_local_name(helper, original) is True


def test_helper_imports_local_name_kwarg():
    # Function with **kwargs: kwarg name tracked as potential local.
    helper = "def _h():\n    import kwargs\n"
    original = "def test(**kwargs):\n    pass\n"
    assert _helper_imports_local_name(helper, original) is True
