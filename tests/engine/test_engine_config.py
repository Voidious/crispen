import textwrap
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.refactors.base import Refactor
from crispen.stats import RunStats


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
    assert s.lines_changed > 0


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
