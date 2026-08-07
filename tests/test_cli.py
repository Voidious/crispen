"""Tests for the CLI entry point."""

import io
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


def test_main_works_when_stdout_lacks_reconfigure():
    """Streams without a ``reconfigure`` method (e.g. plain StringIO, as
    some non-terminal stdout/stderr replacements are) must not break main()."""
    out, err = StringIO(), StringIO()
    assert not hasattr(out, "reconfigure")
    assert not hasattr(err, "reconfigure")
    messages = ["foo.py: IfNotElse: flipped if/else at line 1"]
    with patch("sys.stdin", StringIO("some diff text")):
        with patch("crispen.cli.parse_diff", return_value={"foo.py": [(1, 5)]}):
            with patch("crispen.cli.run_engine", return_value=iter(messages)):
                with patch("sys.stdout", out), patch("sys.stderr", err):
                    main()
    assert "IfNotElse" in out.getvalue()


def test_non_ascii_message_does_not_crash_on_narrow_console_encoding():
    """A cp1252 stdout (Windows default console codepage) must not crash on
    the "→" characters emitted in FileLimiter progress messages."""
    messages = ["crispen: FileLimiter:   → done [1.23s, 100 in / 20 out]"]
    narrow_stdout = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict"
    )
    with patch("sys.stdin", StringIO("some diff text")):
        with patch("crispen.cli.parse_diff", return_value={"foo.py": [(1, 5)]}):
            with patch("crispen.cli.run_engine", return_value=iter(messages)):
                with patch("sys.stdout", narrow_stdout):
                    main()  # must not raise UnicodeEncodeError
    narrow_stdout.seek(0)
    written = narrow_stdout.buffer.getvalue().decode("cp1252")
    assert "done" in written


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
