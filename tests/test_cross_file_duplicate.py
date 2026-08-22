"""Tests for cross_file_duplicate: 100% branch coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from crispen.config import CrispenConfig
from crispen.refactors.cross_file_duplicate import (
    _llm_extract_cross_file,
    _llm_verify_extraction_cross_file,
    run_cross_file_duplicate_extraction,
)
from crispen.stats import RunStats

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Method calls on the `data` parameter only (no free-standing function
# names) — realistic and self-contained, so an extracted helper doesn't
# itself reference something undefined in its own new file. Exactly 3
# statements (weight 3, matching min_duplicate_weight's default): any
# smaller sub-window has weight < 3 and can't itself independently qualify
# as a duplicate — important for skip-marker tests, where only a window
# starting exactly on the marked line is protected.
_DUP_BODY = (
    "    stripped = data.strip()\n"
    '    upper = stripped.replace(" ", "")\n'
    "    return upper\n"
)


def _write_dup_pair(tmp_path: Path, body: str = _DUP_BODY):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    f1.write_text(f"def foo():\n{body}", encoding="utf-8")
    f2.write_text(f"def bar():\n{body}", encoding="utf-8")
    return f1, f2


def _per_file_for(*files: Path) -> dict:
    per_file = {}
    for f in files:
        src = f.read_text(encoding="utf-8")
        per_file[str(f)] = {
            "original": src,
            "source": src,
            "msgs": [],
            "candidates": {},
            "ranges": [(1, len(src.splitlines()))],
        }
    return per_file


def _make_veto_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_extract_response(data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "extract_cross_file_helper"
    block.input = data
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_verify_response(is_correct: bool, issues: list) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "verify_extraction"
    block.input = {"is_correct": is_correct, "issues": issues}
    resp = MagicMock()
    resp.content = [block]
    return resp


_HAPPY_EXTRACT = {
    "function_name": "shared_helper",
    "helper_source": (
        "def shared_helper(data):\n"
        "    stripped = data.strip()\n"
        "    upper = stripped.upper()\n"
        '    parts = upper.split(",")\n'
        "    return parts\n"
    ),
    "call_site_replacements": [
        "    return shared_helper(data)\n",
        "    return shared_helper(data)\n",
    ],
}


def _cfg(tmp_path: Path, **overrides) -> CrispenConfig:
    # min_duplicate_weight uses the class default (3), not 1: with weight 1,
    # each individual "var = func(other_var)" line in _DUP_BODY normalizes
    # identically to every other line in the block, so the single-line
    # windows would themselves look like spurious duplicates and consume
    # LLM mock responses meant for the real 3-statement group.
    defaults: dict = {"extraction_retries": 1, "llm_verify_retries": 1}
    defaults.update(overrides)
    return CrispenConfig(**defaults)


# ---------------------------------------------------------------------------
# Early-return branches
# ---------------------------------------------------------------------------


def test_no_repo_root_returns_immediately(tmp_path):
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    msgs = list(run_cross_file_duplicate_extraction(per_file, None, _cfg(tmp_path)))
    assert msgs == []


def test_single_file_returns_immediately(tmp_path):
    f1, _ = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1)
    msgs = list(
        run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
    )
    assert msgs == []


def test_no_cross_file_group_returns_immediately(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    f1.write_text("def foo():\n    return 1\n", encoding="utf-8")
    f2.write_text("def bar():\n    return 2\n", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    msgs = list(
        run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
    )
    assert msgs == []


def test_parse_error_file_is_skipped_not_fatal(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    f1.write_text("def f(:\n", encoding="utf-8")  # syntax error
    f2.write_text(f"def bar():\n{_DUP_BODY}", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    msgs = list(
        run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
    )
    assert msgs == []


def test_skip_marker_excludes_sequence(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    f1.write_text(
        "def foo():\n"
        "    stripped = data.strip()  # crispen: skip=duplicate_extractor\n"
        '    upper = stripped.replace(" ", "")\n'
        "    return upper\n",
        encoding="utf-8",
    )
    f2.write_text(f"def bar():\n{_DUP_BODY}", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert msgs == []
    mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path + veto/extraction/verify branches
# ---------------------------------------------------------------------------


def test_happy_path_extracts_and_writes_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), verbose=True, stats=stats
            )
        )
    assert len(msgs) == 1
    assert "extracted 'shared_helper'" in msgs[0]
    assert "across 2 files" in msgs[0]
    assert "pkg.common" in msgs[0]

    helper_file = tmp_path / "pkg" / "common.py"
    assert helper_file.exists()
    assert "def shared_helper(data):" in helper_file.read_text(encoding="utf-8")

    for f in (f1, f2):
        new_src = per_file[str(f)]["source"]
        assert "from pkg.common import shared_helper" in new_src
        assert "shared_helper(data)" in new_src
        assert "data.strip()" not in new_src  # original block replaced

    assert stats.duplicate_extracted == 1
    assert stats.llm_veto_calls == 1
    assert stats.llm_edit_calls == 1
    assert stats.llm_verify_calls == 1
    assert str(helper_file) in stats.files_edited


# Regression (found by a live self-check run): a duplicated block that was
# the only user of an import in a file leaves that import dead once the
# block's only use moves into the new cross-file helper. Weight 3 (3
# statements), matching _DUP_BODY's shape.
_DUP_BODY_THREADING = (
    "    t = threading.Thread(target=lambda: None)\n" "    t.start()\n" "    t.join()\n"
)

_THREADING_EXTRACT = {
    "function_name": "shared_helper",
    "helper_source": (
        "import threading\n"
        "\n"
        "\n"
        "def shared_helper():\n"
        "    t = threading.Thread(target=lambda: None)\n"
        "    t.start()\n"
        "    t.join()\n"
    ),
    "call_site_replacements": [
        "    shared_helper()\n",
        "    shared_helper()\n",
    ],
}


def test_dead_import_stripped_after_only_use_extracted(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path, body=_DUP_BODY_THREADING)
    for f, funcname in ((f1, "foo"), (f2, "bar")):
        f.write_text(
            f"import threading\n\n\ndef {funcname}():\n{_DUP_BODY_THREADING}",
            encoding="utf-8",
        )
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_THREADING_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert len(msgs) == 1

    for f in (f1, f2):
        new_src = per_file[str(f)]["source"]
        assert "shared_helper()" in new_src
        assert "import threading" not in new_src

    helper_file = tmp_path / "pkg" / "common.py"
    assert "import threading" in helper_file.read_text(encoding="utf-8")


def test_veto_rejected_skips_group(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_veto_response(
            False, "coincidental"
        )
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), stats=stats
            )
        )
    assert msgs == []
    assert stats.llm_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_veto_api_timeout_skips_group(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.cross_file_duplicate._run_with_timeout",
            side_effect=__import__(
                "crispen.refactors.duplicate_extractor", fromlist=["_ApiTimeout"]
            )._ApiTimeout("timed out"),
        ),
    ):
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert msgs == []


def test_extraction_api_timeout_skips_group(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from crispen.refactors.duplicate_extractor import _ApiTimeout

    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)

    call_count = {"n": 0}

    def _mock_run(func, timeout, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (True, "same op", "")
        raise _ApiTimeout("timed out")

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.cross_file_duplicate._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert msgs == []


def test_extraction_retry_succeeds_second_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["call_site_replacements"] = [
        "    return shared_helper(data)\n"
    ]  # len mismatch
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), stats=stats
            )
        )
    assert len(msgs) == 1
    assert stats.llm_edit_calls == 2


def test_extraction_retries_exhausted(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["call_site_replacements"] = ["    return shared_helper(data)\n"]
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1


def test_verify_rejected_retries_then_exhausts(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(False, ["issue A"]),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(False, ["issue B"]),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, llm_verify_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.llm_rejected == 1
    assert stats.llm_verify_calls == 2


def test_verify_timeout_skips_group(tmp_path, monkeypatch, capsys):
    """Verify timeout must fail closed (skip the group), not silently accept."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from crispen.refactors.duplicate_extractor import _ApiTimeout

    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)

    responses = iter(
        [
            (True, "same op", ""),
            _HAPPY_EXTRACT,
        ]
    )

    def _mock_run(func, timeout, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "_llm_verify_extraction_cross_file":
            raise _ApiTimeout("timed out")
        return next(responses)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.cross_file_duplicate._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert len(msgs) == 0
    err = capsys.readouterr().err
    assert "cross-file API call timed out, skipping group" in err


# ---------------------------------------------------------------------------
# Placement / existing helper file
# ---------------------------------------------------------------------------


def test_appends_to_existing_helper_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    common = tmp_path / "pkg" / "common.py"
    common.write_text("def existing_helper():\n    return 0\n", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert len(msgs) == 1
    content = common.read_text(encoding="utf-8")
    assert "def existing_helper():" in content
    assert "def shared_helper(data):" in content


def test_helper_file_merge_syntax_error_skips_group(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    common = tmp_path / "pkg" / "common.py"
    # Existing content ends mid-statement; appending after it breaks compile.
    common.write_text("x = (\n", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert msgs == []


# ---------------------------------------------------------------------------
# _llm_extract_cross_file / _llm_verify_extraction_cross_file prompt branches
# ---------------------------------------------------------------------------


def _seq(filepath, start, end, scope="foo", source="    x = 1\n"):
    from crispen.refactors.duplicate_extractor import _SeqInfo

    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope=scope,
        source=source,
        fingerprint="",
        filepath=filepath,
    )


def test_llm_extract_cross_file_all_notes():
    client = MagicMock()
    client.messages.create.return_value = _make_extract_response(_HAPPY_EXTRACT)
    group = [_seq("a.py", 1, 2), _seq("b.py", 1, 2)]
    result = _llm_extract_cross_file(
        client,
        group,
        {
            "a.py": "def foo():\n    x = 1\n    y = 2\n",
            "b.py": "def bar():\n    x = 1\n",
        },
        "pkg.common",
        escaping_vars=frozenset({"z"}),
        used_names=frozenset({"other"}),
        helper_docstrings=False,
        veto_notes="watch out",
        prev_failures=["prior failure"],
        prev_output={"helper_source": "def x(): pass", "call_site_replacements": ["a"]},
    )
    assert result == _HAPPY_EXTRACT


def test_llm_extract_cross_file_minimal():
    client = MagicMock()
    client.messages.create.return_value = _make_extract_response(_HAPPY_EXTRACT)
    group = [_seq("a.py", 1, 2), _seq("b.py", 1, 2)]
    result = _llm_extract_cross_file(
        client,
        group,
        {"a.py": "def foo():\n    x = 1\n", "b.py": "def bar():\n    x = 1\n"},
        "pkg.common",
    )
    assert result == _HAPPY_EXTRACT


def test_llm_extract_cross_file_prompt_includes_return_note():
    client = MagicMock()
    client.messages.create.return_value = _make_extract_response(_HAPPY_EXTRACT)
    group = [
        _seq("a.py", 1, 1, source="    return 1\n"),
        _seq("b.py", 1, 1, source="    return 1\n"),
    ]
    _llm_extract_cross_file(
        client,
        group,
        {"a.py": "def foo():\n    return 1\n", "b.py": "def bar():\n    return 1\n"},
        "pkg.common",
    )
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "call site replacement for every occurrence must be" in prompt
    assert "`return <helper_call>(...)`" in prompt


def test_llm_extract_cross_file_prompt_omits_return_note_when_not_applicable():
    client = MagicMock()
    client.messages.create.return_value = _make_extract_response(_HAPPY_EXTRACT)
    group = [_seq("a.py", 1, 2), _seq("b.py", 1, 2)]
    _llm_extract_cross_file(
        client,
        group,
        {"a.py": "def foo():\n    x = 1\n", "b.py": "def bar():\n    x = 1\n"},
        "pkg.common",
    )
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "call site replacement for every occurrence must be" not in prompt


def test_llm_verify_extraction_cross_file():
    client = MagicMock()
    client.messages.create.return_value = _make_verify_response(True, [])
    group = [_seq("a.py", 2, 4), _seq("b.py", 2, 4)]
    is_correct, issues = _llm_verify_extraction_cross_file(
        client,
        group,
        _HAPPY_EXTRACT["helper_source"],
        _HAPPY_EXTRACT["call_site_replacements"],
        {"a.py": f"def foo():\n{_DUP_BODY}", "b.py": f"def bar():\n{_DUP_BODY}"},
    )
    assert is_correct is True
    assert issues == []


def test_llm_verify_extraction_cross_file_rejects_when_truncated():
    client = MagicMock()
    resp = MagicMock()
    resp.content = []  # no tool_use block, e.g. response cut off by max_tokens
    client.messages.create.return_value = resp
    group = [_seq("a.py", 2, 4), _seq("b.py", 2, 4)]
    is_correct, issues = _llm_verify_extraction_cross_file(
        client,
        group,
        _HAPPY_EXTRACT["helper_source"],
        _HAPPY_EXTRACT["call_site_replacements"],
        {"a.py": f"def foo():\n{_DUP_BODY}", "b.py": f"def bar():\n{_DUP_BODY}"},
    )
    assert is_correct is False
    assert issues


# ---------------------------------------------------------------------------
# End-to-end via run_engine
# ---------------------------------------------------------------------------


def test_run_engine_cross_file_extraction_end_to_end(tmp_path, monkeypatch):
    from crispen.engine import run_engine

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    changed = {
        str(f1): [(1, 4)],
        str(f2): [(1, 4)],
    }
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_engine(
                changed,
                _repo_root=str(tmp_path),
                config=_cfg(
                    tmp_path,
                    enabled_refactors=["duplicate_extractor", "match_function"],
                    match_functions_scope="file",
                ),
            )
        )
    assert any("extracted 'shared_helper'" in m for m in msgs)
    helper_file = tmp_path / "pkg" / "common.py"
    assert helper_file.exists()
    assert f1.read_text(encoding="utf-8").count("shared_helper(data)") == 1
    assert f2.read_text(encoding="utf-8").count("shared_helper(data)") == 1


# ---------------------------------------------------------------------------
# verbose=False branches
# ---------------------------------------------------------------------------


def test_happy_path_verbose_false(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), verbose=False
            )
        )
    assert len(msgs) == 1


def test_algorithmic_retry_verbose_false(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["call_site_replacements"] = ["    return shared_helper(data)\n"]
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), verbose=False
            )
        )
    assert len(msgs) == 1


def test_verify_timeout_skips_group_verbose_false(tmp_path, monkeypatch):
    """Verify timeout must fail closed (skip the group), not silently accept."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from crispen.refactors.duplicate_extractor import _ApiTimeout

    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    responses = iter([(True, "same op", ""), _HAPPY_EXTRACT])

    def _mock_run(func, timeout, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "_llm_verify_extraction_cross_file":
            raise _ApiTimeout("timed out")
        return next(responses)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.cross_file_duplicate._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), verbose=False
            )
        )
    assert len(msgs) == 0


def test_verify_retry_verbose_false(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(False, ["issue A"]),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), verbose=False
            )
        )
    assert len(msgs) == 1


def test_helper_merge_syntax_error_verbose_false(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    common = tmp_path / "pkg" / "common.py"
    common.write_text("x = (\n", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path), verbose=False
            )
        )
    assert msgs == []


# ---------------------------------------------------------------------------
# Algorithmic-failure injection branches
# ---------------------------------------------------------------------------


def test_verify_extraction_algorithmic_failure(tmp_path, monkeypatch):
    """Same-length call_replacements, but one is not valid Python."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["call_site_replacements"] = [
        "    return shared_helper(data\n",  # unclosed paren
        "    return shared_helper(data)\n",
    ]
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1


def test_per_file_compile_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    stats = RunStats()
    with (
        patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls,
        patch(
            "crispen.refactors.cross_file_duplicate._lift_and_dedup_imports",
            return_value="def f(:\n",
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_extract_response(_HAPPY_EXTRACT),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1


def test_func_not_called_in_result(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["call_site_replacements"] = [
        "    return 1\n",
        "    return 1\n",
    ]
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1


def test_undefined_name_introduced(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["call_site_replacements"] = [
        "    return shared_helper(totally_undefined_name)\n",
        "    return shared_helper(data)\n",
    ]
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1


def test_helper_references_undefined_private_name(tmp_path, monkeypatch):
    """Regression (found by a live self-check run against crispen's own
    source): a helper_source referencing a name private to the *original*
    file (e.g. a module-level `_llm_client` import or a `_SOME_TOOL`
    constant) compiles fine standalone — compile() doesn't resolve names —
    but crashes with NameError the moment it's actually called from its new,
    separate helper file. Must be caught before writing, not just before
    finding a compile-time syntax error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["helper_source"] = (
        "def shared_helper(data):\n" "    return _some_private_module.do_thing(data)\n"
    )
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_helper_class_collides_with_origin_class(tmp_path, monkeypatch):
    """Regression (found by a live self-check run against crispen's own
    source): the LLM's helper_source defines its own class (e.g. a custom
    exception type used for a hard-timeout guard) with the same name as a
    class an origin file already defines at module level. Both classes
    compile fine standalone and neither name is undefined, so pyflakes-based
    checks miss it -- but they're two distinct objects sharing a name.
    Classes/exceptions match by identity, not name: an origin file's own
    `except _ApiTimeout:` elsewhere in the file would silently stop matching
    whatever the shared helper raises."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    f1.write_text(
        "class _ApiTimeout(Exception):\n    pass\n\n\n"
        + f1.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["helper_source"] = (
        "class _ApiTimeout(Exception):\n    pass\n\n\n"
        + _HAPPY_EXTRACT["helper_source"]
    )
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_helper_class_no_collision_with_origin_class(tmp_path, monkeypatch):
    """A helper class whose name doesn't collide with anything an origin
    file already defines is unaffected by the new check."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    f1.write_text(
        "class _SomeOtherThing(Exception):\n    pass\n\n\n"
        + f1.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    per_file = _per_file_for(f1, f2)
    extract = dict(_HAPPY_EXTRACT)
    extract["helper_source"] = (
        "class _ApiTimeout(Exception):\n    pass\n\n\n"
        + _HAPPY_EXTRACT["helper_source"]
    )
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(extract),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert len(msgs) == 1
    assert stats.algorithmic_rejected == 0
    assert (tmp_path / "pkg" / "common.py").exists()


def test_helper_imports_class_orphaning_sibling_origin_rejected(tmp_path, monkeypatch):
    """Regression (found by a live self-check run against crispen's own
    source): both origin files define their own, separate `_ApiTimeout`
    class for the same purpose. The helper correctly avoids *redefining*
    either one (that's the previous check's job) but instead *imports* only
    one origin's class -- silently orphaning the other origin file's own,
    un-extracted `except _ApiTimeout:` (or similar) references, which still
    expect their own distinct class object. Classes/exceptions match by
    identity, not name."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    f1.write_text(
        "class _ApiTimeout(Exception):\n    pass\n\n\n"
        + f1.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    f2.write_text(
        "class _ApiTimeout(Exception):\n"
        "    pass\n\n\n"
        "def other():\n"
        "    try:\n"
        "        pass\n"
        "    except _ApiTimeout:\n"
        "        pass\n\n\n" + f2.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["helper_source"] = (
        "from pkg.a import _ApiTimeout\n\n\n"
        "def shared_helper(data):\n"
        "    if not data:\n"
        "        raise _ApiTimeout('empty')\n"
        "    stripped = data.strip()\n"
        "    upper = stripped.upper()\n"
        '    parts = upper.split(",")\n'
        "    return parts\n"
    )
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_helper_imports_class_no_orphaned_sibling_origin(tmp_path, monkeypatch):
    """A helper importing a class from one origin file is unaffected by the
    new check when no *different* origin file defines its own separate,
    same-named class."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    f1.write_text(
        "class _ApiTimeout(Exception):\n    pass\n\n\n"
        + f1.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    per_file = _per_file_for(f1, f2)
    extract = dict(_HAPPY_EXTRACT)
    extract["helper_source"] = (
        "from pkg.a import _ApiTimeout\n\n\n"
        "def shared_helper(data):\n"
        "    if not data:\n"
        "        raise _ApiTimeout('empty')\n"
        "    stripped = data.strip()\n"
        "    upper = stripped.upper()\n"
        '    parts = upper.split(",")\n'
        "    return parts\n"
    )
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(extract),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert len(msgs) == 1
    assert stats.algorithmic_rejected == 0
    assert (tmp_path / "pkg" / "common.py").exists()


def test_dropped_directive_comment_rejected(tmp_path, monkeypatch):
    """Regression (same shape as a live self-check finding, reproduced here
    at the cross-file layer): the original duplicate block ends with a line
    carrying a `# pragma: no cover` comment, but the extracted helper drops
    the comment while keeping the guarded code. Every test still passes --
    the drop is syntactically invisible -- but it silently reintroduces
    whatever the comment was suppressing (here, a coverage-gate failure)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(
        tmp_path,
        body=(
            "    stripped = data.strip()\n"
            '    upper = stripped.replace(" ", "")\n'
            "    return upper  # pragma: no cover\n"
        ),
    )
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["helper_source"] = (
        "def shared_helper(data):\n"
        "    stripped = data.strip()\n"
        '    upper = stripped.replace(" ", "")\n'
        "    return upper\n"
    )
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_directive_comment_preserved_not_flagged(tmp_path, monkeypatch):
    """A helper that keeps the original block's directive comment is
    unaffected by the check."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(
        tmp_path,
        body=(
            "    stripped = data.strip()\n"
            '    upper = stripped.replace(" ", "")\n'
            "    return upper  # pragma: no cover\n"
        ),
    )
    per_file = _per_file_for(f1, f2)
    extract = dict(_HAPPY_EXTRACT)
    extract["helper_source"] = (
        "def shared_helper(data):\n"
        "    stripped = data.strip()\n"
        "    upper = stripped.upper()\n"
        '    parts = upper.split(",")\n'
        "    return parts  # pragma: no cover\n"
    )
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(extract),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert len(msgs) == 1
    assert stats.algorithmic_rejected == 0
    assert (tmp_path / "pkg" / "common.py").exists()


_DEFAULT_GLOBAL_BODY = (
    "    stripped = data.strip()\n"
    '    upper = stripped.replace(" ", "")\n'
    "    print(upper, file=sys.stderr, flush=True)\n"
)


def test_default_param_drops_call_time_global_rejected(tmp_path, monkeypatch):
    """Regression (same shape as a live self-check finding, reproduced here
    at the cross-file layer): both original call sites passed
    ``file=sys.stderr`` explicitly, but the extracted helper turns it into a
    default parameter value and drops it from the call sites. A default is
    bound once at def-time, not fresh on each call -- invisible to every
    other check since it's semantically identical unless something (e.g.
    pytest's capsys fixture) reassigns sys.stderr after the helper is
    defined."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path, body=_DEFAULT_GLOBAL_BODY)
    per_file = _per_file_for(f1, f2)
    bad_extract = dict(_HAPPY_EXTRACT)
    bad_extract["helper_source"] = (
        "import sys\n\n\n"
        "def shared_helper(data, file=sys.stderr):\n"
        "    stripped = data.strip()\n"
        '    upper = stripped.replace(" ", "")\n'
        "    print(upper, file=file, flush=True)\n"
    )
    bad_extract["call_site_replacements"] = [
        "    shared_helper(data)\n",
        "    shared_helper(data)\n",
    ]
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_default_param_global_kept_explicit_not_flagged(tmp_path, monkeypatch):
    """A helper that still receives the call-time global explicitly at each
    call site (not relying on the new default) is unaffected by the check."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path, body=_DEFAULT_GLOBAL_BODY)
    per_file = _per_file_for(f1, f2)
    extract = dict(_HAPPY_EXTRACT)
    extract["helper_source"] = (
        "import sys\n\n\n"
        "def shared_helper(data, file=sys.stderr):\n"
        "    stripped = data.strip()\n"
        '    upper = stripped.replace(" ", "")\n'
        "    print(upper, file=file, flush=True)\n"
    )
    extract["call_site_replacements"] = [
        "    shared_helper(data, file=sys.stderr)\n",
        "    shared_helper(data, file=sys.stderr)\n",
    ]
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(extract),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert len(msgs) == 1
    assert stats.algorithmic_rejected == 0
    assert (tmp_path / "pkg" / "common.py").exists()


# ---------------------------------------------------------------------------
# Integration: dropped-escaping-var-capture guard (shared with same-file)
# ---------------------------------------------------------------------------


def _write_escape_capture_pair(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    # A module-level _rl_delay makes the name always resolvable (matching the
    # real bug's shape: an outer-scope value that a dropped call-site
    # reassignment leaves silently stale, not a hard undefined-name error).
    f1.write_text(
        "_rl_delay = 1\n\n\n"
        "def foo(a):\n"
        "    x = a + 1\n"
        "    y = x * 2\n"
        "    _rl_delay = x + y\n"
        "    if _rl_delay is not None:\n"
        "        wait(_rl_delay)\n",
        encoding="utf-8",
    )
    f2.write_text(
        "_rl_delay = 1\n\n\n"
        "def bar(a):\n"
        "    x = a + 1\n"
        "    y = x * 2\n"
        "    _rl_delay = x + y\n"
        "    log(_rl_delay)\n",
        encoding="utf-8",
    )
    return f1, f2


def test_dropped_escaping_var_capture_rejected(tmp_path, monkeypatch):
    """Regression (same shape as a live self-check finding, reproduced here
    at the cross-file layer): one call site drops the reassignment of a
    variable that escapes for its own occurrence."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_escape_capture_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = {
        "function_name": "shared_helper",
        "helper_source": (
            "def shared_helper(a):\n"
            "    x = a + 1\n"
            "    y = x * 2\n"
            "    return x + y\n"
        ),
        "call_site_replacements": [
            "    _rl_delay = shared_helper(a)\n",
            "    shared_helper(a)\n",  # drops the reassignment
        ],
    }
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_escaping_var_capture_preserved_not_flagged_cross_file(tmp_path, monkeypatch):
    """Both call sites capture the return value -- unaffected by the check."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_escape_capture_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    extract = {
        "function_name": "shared_helper",
        "helper_source": (
            "def shared_helper(a):\n"
            "    x = a + 1\n"
            "    y = x * 2\n"
            "    return x + y\n"
        ),
        "call_site_replacements": [
            "    _rl_delay = shared_helper(a)\n",
            "    _rl_delay = shared_helper(a)\n",
        ],
    }
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(extract),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert len(msgs) == 1
    assert stats.algorithmic_rejected == 0
    assert (tmp_path / "pkg" / "common.py").exists()


# ---------------------------------------------------------------------------
# Integration: call-site argument identity mismatch guard (shared with
# same-file)
# ---------------------------------------------------------------------------


def _write_crossed_args_pair(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    # A single tuple-assignment statement (weight 1, below min_duplicate_weight)
    # so this shared setup isn't itself detected as a spurious duplicate group.
    globals_block = 'trial_deps, candidate, file_deps, chosen = {}, "a", {}, "b"\n\n\n'
    f1.write_text(
        globals_block + "def trial_step():\n"
        "    trial_deps[candidate] = None\n"
        "    depth = topo_depth(trial_deps, candidate)\n"
        "    depth += 1\n"
        "    if depth > 0:\n"
        "        log_trial(depth)\n",
        encoding="utf-8",
    )
    f2.write_text(
        globals_block + "def apply_step():\n"
        "    file_deps[chosen] = None\n"
        "    depth = topo_depth(file_deps, chosen)\n"
        "    depth += 1\n"
        "    log_apply(depth)\n",
        encoding="utf-8",
    )
    return f1, f2


def test_call_site_argument_identity_mismatch_rejected(tmp_path, monkeypatch):
    """Regression (same shape as a live self-check finding, reproduced here
    at the cross-file layer): the two call sites' arguments got crossed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_crossed_args_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    bad_extract = {
        "function_name": "shared_register",
        "helper_source": (
            "def shared_register(deps, key):\n"
            "    deps[key] = None\n"
            "    depth = len(deps)\n"
            "    return depth + 1\n"
        ),
        "call_site_replacements": [
            "    depth = shared_register(file_deps, chosen)\n",
            "    depth = shared_register(trial_deps, candidate)\n",
        ],
    }
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(bad_extract),
            _make_extract_response(bad_extract),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert msgs == []
    assert stats.algorithmic_rejected == 1
    assert not (tmp_path / "pkg" / "common.py").exists()


def test_call_site_argument_identity_not_crossed_not_flagged_cross_file(
    tmp_path, monkeypatch
):
    """Each call site's arguments come from its own block -- unaffected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_crossed_args_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    extract = {
        "function_name": "shared_register",
        "helper_source": (
            "def shared_register(deps, key):\n"
            "    deps[key] = None\n"
            "    depth = len(deps)\n"
            "    return depth + 1\n"
        ),
        "call_site_replacements": [
            "    depth = shared_register(trial_deps, candidate)\n",
            "    depth = shared_register(file_deps, chosen)\n",
        ],
    }
    stats = RunStats()
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(extract),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file,
                str(tmp_path),
                _cfg(tmp_path, extraction_retries=1),
                stats=stats,
            )
        )
    assert len(msgs) == 1
    assert stats.algorithmic_rejected == 0
    assert (tmp_path / "pkg" / "common.py").exists()


def test_helper_docstrings_true_keeps_docstring(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    docstring_extract = dict(_HAPPY_EXTRACT)
    docstring_extract["helper_source"] = (
        "def shared_helper(data):\n"
        '    """Do the shared thing."""\n'
        "    stripped = data.strip()\n"
        "    upper = stripped.upper()\n"
        '    parts = upper.split(",")\n'
        "    return parts\n"
    )
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(docstring_extract),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), _cfg(tmp_path, helper_docstrings=True)
            )
        )
    assert len(msgs) == 1
    content = (tmp_path / "pkg" / "common.py").read_text(encoding="utf-8")
    assert "Do the shared thing." in content


def test_blank_line_after_block_skips_hint(tmp_path, monkeypatch):
    """A blank line right after the block means no 'line after' hint to add
    (as opposed to real trailing content, which does add one)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    # Blank line immediately after the block, then more content — exercises
    # next_idx < len(src_lines) (True) together with next_line.strip() (False).
    f1.write_text(
        f"def foo():\n{_DUP_BODY}\ndef after():\n    pass\n", encoding="utf-8"
    )
    f2.write_text(f"def bar():\n{_DUP_BODY}", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert len(msgs) == 1


def test_all_calls_succeed_via_mock_no_timing(tmp_path, monkeypatch):
    """_timing_out stays empty when _run_with_timeout is mocked to return
    directly, without ever touching the _timing_out kwarg it's normally
    responsible for populating (the real function does that internally)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    f1, f2 = _write_dup_pair(tmp_path)
    per_file = _per_file_for(f1, f2)
    responses = iter([(True, "same op", ""), _HAPPY_EXTRACT, (True, [])])

    def _mock_run(func, timeout, *args, **kwargs):
        return next(responses)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.cross_file_duplicate._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert len(msgs) == 1


def test_creates_init_py_for_new_package(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # No pkg/__init__.py this time — the common ancestor dir needs one created.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    f1.write_text(f"def foo():\n{_DUP_BODY}", encoding="utf-8")
    f2.write_text(f"def bar():\n{_DUP_BODY}", encoding="utf-8")
    per_file = _per_file_for(f1, f2)
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same op"),
            _make_extract_response(_HAPPY_EXTRACT),
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(per_file, str(tmp_path), _cfg(tmp_path))
        )
    assert len(msgs) == 1
    assert (pkg / "__init__.py").exists()
