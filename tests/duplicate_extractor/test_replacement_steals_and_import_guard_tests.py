from crispen.refactors.duplicate_extractor import (
    _replacement_steals_post_block_line,
    _helper_imports_local_name,
)
from .replacement_steals_and_import_guard_tests import _make_steal_seq


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
