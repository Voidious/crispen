"""Tests for the CLI entry point."""

from io import StringIO
from unittest.mock import patch

import pytest

from crispen.cli import main


def test_empty_stdin_exits_1(capsys):
    with patch("sys.stdin", StringIO("")):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1
    assert "no diff provided" in capsys.readouterr().err


def test_whitespace_stdin_exits_1():
    with patch("sys.stdin", StringIO("   \n  ")):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1


def test_diff_with_no_changed_files(capsys):
    with patch("sys.stdin", StringIO("some diff text")):
        with patch("crispen.cli.parse_diff", return_value={}):
            main()
    assert capsys.readouterr().out == ""


def test_diff_prints_engine_messages(capsys):
    messages = ["foo.py: IfNotElse: flipped if/else at line 1"]
    with patch("sys.stdin", StringIO("some diff text")):
        with patch("crispen.cli.parse_diff", return_value={"foo.py": [(1, 5)]}):
            with patch("crispen.cli.run_engine", return_value=iter(messages)):
                main()
    assert "IfNotElse" in capsys.readouterr().out


def test_main_no_summary_when_no_llm_calls_and_no_edits(capsys):
    """Summary is suppressed when no LLM calls and no edits were made."""
    with patch("sys.stdin", StringIO("some diff text")):
        with patch("crispen.cli.parse_diff", return_value={"foo.py": [(1, 5)]}):
            with patch("crispen.cli.run_engine", return_value=iter([])):
                main()
    assert capsys.readouterr().out == ""


def test_main_prints_summary_when_llm_calls_but_no_edits(capsys):
    """Summary is printed when LLM calls were made even if no edits resulted."""

    def fake_engine(changed, config, stats=None, **kwargs):
        if stats is not None:
            stats.llm_veto_calls = 1
            stats.llm_rejected = 1
        return iter([])

    with patch("sys.stdin", StringIO("some diff text")):
        with patch("crispen.cli.parse_diff", return_value={"foo.py": [(1, 5)]}):
            with patch("crispen.cli.run_engine", side_effect=fake_engine):
                main()
    assert "--- crispen summary ---" in capsys.readouterr().out


def test_main_prints_summary(capsys):
    """Summary is printed after all engine messages."""

    def fake_engine(changed, config, stats=None, **kwargs):
        if stats is not None:
            stats.if_not_else = 2
            stats.files_edited.append("foo.py")
            stats.lines_added = 3
            stats.lines_deleted = 1
        return iter(["foo.py: IfNotElse: flipped if/else at line 1"])

    with patch("sys.stdin", StringIO("some diff text")):
        with patch("crispen.cli.parse_diff", return_value={"foo.py": [(1, 5)]}):
            with patch("crispen.cli.run_engine", side_effect=fake_engine):
                main()
    out = capsys.readouterr().out
    assert "--- crispen summary ---" in out
    assert "if not/else:" in out
    assert "files edited (1): foo.py" in out
    assert "lines added:           3" in out
    assert "lines deleted:         1" in out
