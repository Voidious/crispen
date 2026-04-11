from unittest.mock import patch
import textwrap
from crispen.config import CrispenConfig
from crispen.engine import _LLM_REFACTOR_KEYS, run_engine
from crispen.file_limiter.runner import FileLimiterResult
from crispen.refactors.base import Refactor
from crispen.stats import RunStats
from .test_engine_file_limiter import _FL_PATCH


def test_config_header_printed_when_llm_refactors_enabled(tmp_path, capsys):
    f = tmp_path / "simple.py"
    f.write_text("x = 1\n", encoding="utf-8")
    list(run_engine({str(f): [(1, 1)]}, config=CrispenConfig()))
    err = capsys.readouterr().err
    assert "--- crispen ---" in err
    assert "provider:" in err
    assert "model:" in err


def test_config_header_suppressed_when_all_llm_refactors_disabled(tmp_path, capsys):
    f = tmp_path / "simple.py"
    f.write_text("x = 1\n", encoding="utf-8")
    cfg = CrispenConfig(disabled_refactors=list(_LLM_REFACTOR_KEYS))
    list(run_engine({str(f): [(1, 1)]}, config=cfg))
    assert "--- crispen ---" not in capsys.readouterr().err


def test_config_header_suppressed_when_changed_empty(capsys):
    list(run_engine({}, config=CrispenConfig()))
    assert "--- crispen ---" not in capsys.readouterr().err


def test_run_engine_accepts_explicit_config(tmp_path):
    """run_engine works when config is provided explicitly."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    config = CrispenConfig()
    msgs = list(run_engine({str(f): [(1, 1)]}, config=config))
    assert msgs == []


def test_run_engine_config_none_loads_default(tmp_path):
    """run_engine with config=None (default) loads config from disk."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    # config=None triggers load_config() internally
    msgs = list(run_engine({str(f): [(1, 1)]}, config=None))
    assert msgs == []


def test_run_engine_stats_populated(tmp_path):
    source = "if not x:\n    a()\nelse:\n    b()\n"
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    s = RunStats()
    list(run_engine({str(f): [(1, 4)]}, config=CrispenConfig(), stats=s))
    assert s.if_not_else == 1
    assert s.files_edited == [str(f)]
    assert s.lines_added + s.lines_deleted > 0


def test_run_engine_stats_none_default(tmp_path):
    """When stats is None (default), engine runs without error."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = list(run_engine({str(f): [(1, 1)]}, config=CrispenConfig()))
    assert msgs == []


def test_engine_disabled_refactors_skips_if_not_else(tmp_path):
    """With if_not_else disabled the pattern is left unchanged."""
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = list(
        run_engine(
            {str(f): [(1, 4)]},
            config=CrispenConfig(disabled_refactors=["if_not_else"]),
        )
    )
    assert not any("IfNotElse" in m for m in msgs)
    assert f.read_text(encoding="utf-8") == source


def test_engine_enabled_refactors_runs_only_listed(tmp_path):
    """enabled_refactors=["if_not_else"] — other refactors don't touch the file."""
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")

    called = []

    class _Spy(Refactor):
        @classmethod
        def name(cls):
            return "Spy"

        def get_changes(self):
            called.append("Spy")
            return []

    with patch("crispen.engine._REFACTORS", [_Spy]):
        with patch("crispen.engine._REFACTOR_KEY", {_Spy: "spy"}):
            list(
                run_engine(
                    {str(f): [(1, 4)]},
                    config=CrispenConfig(enabled_refactors=["if_not_else"]),
                )
            )

    # _Spy is not in enabled_refactors, so it must not have been called.
    assert called == []


def test_engine_file_limiter_skipped_when_disabled(tmp_path):
    """file_limiter in disabled_refactors prevents FileLimiter from running."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    success_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "# new\n"},
        messages=["FileLimiter: moved"],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=success_result) as mock_fl:
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    disabled_refactors=["file_limiter"],
                ),
            )
        )
    mock_fl.assert_not_called()


def test_engine_match_function_disabled_passes_flag_to_duplicate_extractor(tmp_path):
    """disabled_refactors=["match_function"] passes match_functions=False to DE."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    constructed_with: dict = {}

    original_init = __import__(
        "crispen.refactors.duplicate_extractor", fromlist=["DuplicateExtractor"]
    ).DuplicateExtractor.__init__

    def _spy_init(self, *args, **kwargs):
        constructed_with.update(kwargs)
        original_init(self, *args, **kwargs)

    with patch(
        "crispen.engine.caller_detection.DuplicateExtractor.__init__",
        side_effect=_spy_init,
    ):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(disabled_refactors=["match_function"]),
            )
        )

    assert constructed_with.get("match_functions") is False


def test_engine_match_function_enabled_by_default(tmp_path):
    """Without any filter, match_functions=True is passed to DuplicateExtractor."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    constructed_with: dict = {}

    original_init = __import__(
        "crispen.refactors.duplicate_extractor", fromlist=["DuplicateExtractor"]
    ).DuplicateExtractor.__init__

    def _spy_init(self, *args, **kwargs):
        constructed_with.update(kwargs)
        original_init(self, *args, **kwargs)

    with patch(
        "crispen.engine.caller_detection.DuplicateExtractor.__init__",
        side_effect=_spy_init,
    ):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(),
            )
        )

    assert constructed_with.get("match_functions") is True


def test_file_limiter_llm_timing_recorded_in_stats(tmp_path):
    """When FileLimiterResult has llm_elapsed > 0, record_llm_call is invoked."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    timed_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "# new\n"},
        messages=[f"{f}: FileLimiter: moved → utils.py"],
        abort=False,
        llm_elapsed=1.5,
        llm_input_tokens=100,
        llm_output_tokens=50,
    )
    stats = RunStats()
    with patch(_FL_PATCH, return_value=timed_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
                stats=stats,
            )
        )
    assert "file_limiter" in stats.llm_elapsed_by_category


def test_file_limiter_recursive_llm_timing_recorded_in_stats(tmp_path):
    """Recursive FileLimiterResult with llm_elapsed > 0 triggers record_llm_call."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    first_result = FileLimiterResult(
        original_source="# reduced original\n",
        new_files={"chunk.py": "".join(f"x_{i} = {i}\n" for i in range(10))},
        messages=[f"{f}: moved vars → chunk.py"],
        abort=False,
    )
    second_result = FileLimiterResult(
        original_source="# reduced chunk\n",
        new_files={"chunk_a.py": "# a\n"},
        messages=[],
        abort=False,
        llm_elapsed=2.0,
        llm_input_tokens=200,
        llm_output_tokens=80,
    )
    stats = RunStats()
    call_count = 0

    def _fl_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        return first_result if call_count == 1 else second_result

    with patch(_FL_PATCH, side_effect=_fl_side_effect):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
                stats=stats,
            )
        )
    assert "file_limiter" in stats.llm_elapsed_by_category
