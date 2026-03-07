import textwrap
import threading
from unittest.mock import patch
import pytest
from crispen.config import CrispenConfig
from crispen.engine import (
    _apply_tuple_dataclass,
    _blocked_private_scopes,
    _categorize_into_stats,
    _has_callers_outside_ranges,
    _should_run,
    _visit_with_timeout,
    run_engine,
)
from crispen.errors import CrispenAPIError
from crispen.refactors.base import Refactor
from crispen.stats import RunStats


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


def test_skip_missing_file(tmp_path):
    missing = str(tmp_path / "nonexistent.py")
    msgs = _run({missing: [(1, 10)]})
    assert len(msgs) == 1
    assert "SKIP" in msgs[0]
    assert "file not found" in msgs[0]


def test_no_changes_no_messages(tmp_path):
    f = tmp_path / "simple.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 1)]})
    assert msgs == []


def test_applies_refactor_and_writes(tmp_path):
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
    msgs = _run({str(f): [(1, 4)]})
    assert any("IfNotElse" in m for m in msgs)
    assert "if x:" in f.read_text(encoding="utf-8")


def test_rewritten_source_used_when_available(tmp_path):
    """get_rewritten_source() is preferred over new_tree.code when non-None."""
    rewritten = "x = 999  # rewritten\n"

    class _RewritingRefactor(Refactor):
        @classmethod
        def name(cls):
            return "Rewriter"

        def get_rewritten_source(self):
            return rewritten

        def get_changes(self):
            return ["Rewriter: rewrote the file"]

    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RewritingRefactor]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("Rewriter" in m for m in msgs)
    assert f.read_text(encoding="utf-8") == rewritten


def test_skip_parse_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 2)]})
    assert any("parse error" in m for m in msgs)


class _RaisingTransformer(Refactor):
    """A Refactor subclass that always raises during tree traversal."""

    @classmethod
    def name(cls):
        return "RaisingRefactor"

    def leave_Module(self, original_node, updated_node):
        raise RuntimeError("intentional transform error")


def test_skip_transform_error(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RaisingTransformer]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("transform error" in m for m in msgs)


class _CrispenApiErrorRefactor(Refactor):
    @classmethod
    def name(cls):
        return "ApiErrorRefactor"

    def leave_Module(self, original_node, updated_node):
        raise CrispenAPIError("test api error")


def test_crispen_api_error_propagates(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_CrispenApiErrorRefactor]):
        with pytest.raises(CrispenAPIError):
            list(run_engine({str(f): [(1, 1)]}))


def test_tuple_dataclass_transform_error_handled(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    class _FailingTD:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated TupleDataclass failure")

    with patch("crispen.engine.TupleDataclass", _FailingTD):
        msgs = _run({str(f): [(1, 1)]})
    assert any("TupleDataclass" in m and "transform error" in m for m in msgs)


def test_apply_tuple_dataclass_parse_error():
    bad_source = "def f(:\n    pass\n"
    source_out, msgs, td = _apply_tuple_dataclass(
        "fake.py", [(1, 10)], bad_source, False, set()
    )
    assert any("parse error" in m for m in msgs)
    assert td is None
    assert source_out == bad_source


def test_apply_tuple_dataclass_crispen_api_error():
    with patch("crispen.engine.MetadataWrapper") as MockWrapper:
        MockWrapper.return_value.visit.side_effect = CrispenAPIError("test api error")
        with pytest.raises(CrispenAPIError):
            _apply_tuple_dataclass("f.py", [(1, 1)], "x = 1\n", False, set())


def test_visit_with_timeout_completes():
    """Fast visit completes within timeout → returns True."""
    from unittest.mock import MagicMock

    wrapper = MagicMock()
    finder = MagicMock()
    assert _visit_with_timeout(wrapper, finder, 5.0) is True
    wrapper.visit.assert_called_once_with(finder)


def test_visit_with_timeout_fires():
    """Slow visit that never completes → returns False after timeout."""
    block = threading.Event()

    class _HangWrapper:
        def visit(self, finder):
            block.wait()  # blocks until released

    result = _visit_with_timeout(_HangWrapper(), object(), 0.01)
    block.set()  # unblock the daemon thread for cleanup
    assert result is False


def test_phase1_private_caller_updated(tmp_path):
    """Private function callers in the same file are updated after Phase 1."""
    source = textwrap.dedent(
        """\
        def _make_result():
            return (1, 2, 3)

        def use_it():
            a, b, c = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = _run({str(f): [(1, 100)]})
    result = f.read_text(encoding="utf-8")
    assert "_ = _make_result()" in result
    assert any("CallerUpdater" in m for m in msgs)


def test_phase1_private_no_callers_no_caller_updater_msg(tmp_path):
    """Private transform with no callers produces no CallerUpdater message."""
    source = "def _make_result():\n    return (1, 2, 3)\n"
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = _run({str(f): [(1, 100)]})
    assert any("TupleDataclass" in m for m in msgs)
    assert not any("CallerUpdater" in m for m in msgs)


def test_phase1_private_caller_updater_exception_ignored(tmp_path):
    """If CallerUpdater raises during Phase 1, the engine continues gracefully."""
    source = textwrap.dedent(
        """\
        def _make_result():
            return (1, 2, 3)

        def use_it():
            a, b, c = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    with patch("crispen.engine.CallerUpdater", side_effect=RuntimeError("fail")):
        msgs = _run({str(f): [(1, 100)]})
    # TupleDataclass still ran successfully
    assert any("TupleDataclass" in m for m in msgs)


def test_has_callers_outside_ranges_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range is only line 1
    assert _has_callers_outside_ranges(source, "f", [(1, 1)]) is True


def test_has_callers_outside_ranges_not_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range covers line 2
    assert _has_callers_outside_ranges(source, "f", [(1, 2)]) is False


def test_has_callers_outside_ranges_syntax_error():
    assert _has_callers_outside_ranges("def f(:", "f", [(1, 1)]) is False


def test_blocked_private_scopes_finds_outside_callers():
    # _helper called at line 3, diff range only covers line 1
    source = "def _helper(): pass\n\n_helper()\n"
    blocked = _blocked_private_scopes(source, [(1, 1)])
    assert "_helper" in blocked


def test_blocked_private_scopes_ignores_in_range_callers():
    # _helper called at line 3, diff range covers line 3
    source = "def _helper(): pass\n\n_helper()\n"
    blocked = _blocked_private_scopes(source, [(1, 3)])
    assert "_helper" not in blocked


def test_blocked_private_scopes_syntax_error():
    blocked = _blocked_private_scopes("def f(:", [(1, 1)])
    assert blocked == set()


def test_blocked_private_scopes_ignores_public():
    # Public functions (no leading _) should not appear in blocked set
    source = "def helper(): pass\n\nhelper()\n"
    blocked = _blocked_private_scopes(source, [(1, 1)])
    assert "helper" not in blocked


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


def test_categorize_if_not_else():
    s = RunStats()
    _categorize_into_stats(s, "IfNotElse: flipped if/else at line 3")
    assert s.if_not_else == 1
    assert s.total_edits == 1


def test_categorize_tuple_to_dataclass():
    s = RunStats()
    _categorize_into_stats(
        s, "TupleDataclass: replaced 3-tuple with FooResult at line 5"
    )
    assert s.tuple_to_dataclass == 1


def test_categorize_duplicate_matched():
    s = RunStats()
    _categorize_into_stats(s, "DuplicateExtractor: replaced '_f' body with call to 'g'")
    assert s.duplicate_matched == 1
    assert s.duplicate_extracted == 0


def test_categorize_duplicate_extracted():
    s = RunStats()
    _categorize_into_stats(
        s, "DuplicateExtractor: extracted '_helper' from 2 duplicate blocks"
    )
    assert s.duplicate_extracted == 1
    assert s.duplicate_matched == 0


def test_categorize_function_split():
    s = RunStats()
    _categorize_into_stats(s, "split 'big_func': extracted _step_two")
    assert s.function_split == 1


def test_categorize_other_message_ignored():
    s = RunStats()
    _categorize_into_stats(s, "CallerUpdater: expanded FooResult unpacking at line 7")
    assert s.total_edits == 0


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
