import textwrap
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import _should_run, run_engine
from crispen.file_limiter.runner import FileLimiterResult
from crispen.refactors.base import Refactor
from .test_engine_file_limiter import _FL_PATCH


def test_should_run_defaults_allow_all():
    cfg = CrispenConfig()
    for name in (
        "if_not_else",
        "duplicate_extractor",
        "function_splitter",
        "tuple_dataclass",
        "file_limiter",
    ):
        assert _should_run(name, cfg) is True


def test_should_run_enabled_list_allows_listed():
    cfg = CrispenConfig(enabled_refactors=["if_not_else", "function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("function_splitter", cfg) is True


def test_should_run_enabled_list_blocks_unlisted():
    cfg = CrispenConfig(enabled_refactors=["if_not_else"])
    assert _should_run("duplicate_extractor", cfg) is False
    assert _should_run("tuple_dataclass", cfg) is False
    assert _should_run("file_limiter", cfg) is False


def test_should_run_disabled_list_blocks_listed():
    cfg = CrispenConfig(disabled_refactors=["function_splitter", "file_limiter"])
    assert _should_run("function_splitter", cfg) is False
    assert _should_run("file_limiter", cfg) is False


def test_should_run_disabled_list_allows_unlisted():
    cfg = CrispenConfig(disabled_refactors=["function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("tuple_dataclass", cfg) is True


def test_should_run_enabled_takes_precedence_over_disabled():
    # enabled_refactors non-empty → disabled_refactors is ignored
    cfg = CrispenConfig(
        enabled_refactors=["if_not_else"],
        disabled_refactors=["if_not_else"],
    )
    assert _should_run("if_not_else", cfg) is True


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

    with patch("crispen.engine.DuplicateExtractor.__init__", side_effect=_spy_init):
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

    with patch("crispen.engine.DuplicateExtractor.__init__", side_effect=_spy_init):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(),
            )
        )

    assert constructed_with.get("match_functions") is True
