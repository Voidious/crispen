from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import (
    _EXCLUDED_DIR_NAMES,
    _blocked_private_scopes,
    _build_alias_map,
    _categorize_into_stats,
    _compute_qname,
    _file_to_module,
    _find_outside_callers,
    _find_repo_root,
    _has_callers_outside_ranges,
    _should_run,
    run_engine,
)
from crispen.refactors.base import Refactor
from crispen.stats import RunStats
import textwrap


def test_find_repo_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src"
    subdir.mkdir()
    f = subdir / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root == str(tmp_path)


def test_find_repo_root_not_found(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root is None


def test_file_to_module_regular_file(tmp_path):
    f = tmp_path / "mypkg" / "service.py"
    f.parent.mkdir()
    f.write_text("x = 1\n")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg.service"


def test_file_to_module_init(tmp_path):
    f = tmp_path / "mypkg" / "__init__.py"
    f.parent.mkdir()
    f.write_text("")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg"


def test_compute_qname(tmp_path):
    f = tmp_path / "pkg" / "mod.py"
    f.parent.mkdir()
    f.write_text("")
    assert _compute_qname(str(tmp_path), str(f), "my_func") == "pkg.mod.my_func"


def test_build_alias_map_identity_only(tmp_path):
    # No __init__.py in tmp_path → only identity mapping returned.
    alias_map = _build_alias_map(str(tmp_path), {"a.b.func"})
    assert alias_map == {"a.b.func": "a.b.func"}


def test_build_alias_map_with_reexport(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert "mypkg.get_user" in alias_map
    assert alias_map["mypkg.get_user"] == "mypkg.service.get_user"


def test_build_alias_map_star_import_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import *\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Star import does not create an alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_ambiguous_name_skipped(tmp_path):
    # Two canonical qnames share the same function name → alias is ambiguous.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(
        str(tmp_path),
        {"mypkg.service.get_user", "mypkg.other.get_user"},
    )
    # Ambiguous: skip adding the alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_invalid_init_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def f(:\n    pass\n")  # invalid Python
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Gracefully skips the unreadable __init__.py
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_find_outside_callers_empty_qnames(tmp_path):
    result = _find_outside_callers(str(tmp_path), set(), set())
    assert result == set()


def test_find_outside_callers_no_outside_py_files(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    result = _find_outside_callers(str(tmp_path), {"pkg.func"}, {str(f.resolve())})
    # All .py files are in the diff → nothing to scan outside
    assert result == set()


def test_find_outside_callers_finds_caller(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    service = pkg / "service.py"
    service.write_text("def get_user():\n    return (1, 2, 3)\n")
    outside = tmp_path / "outside.py"
    outside.write_text("from mypkg.service import get_user\nget_user()\n")

    qname = "mypkg.service.get_user"
    diff_files = {str(service.resolve())}
    result = _find_outside_callers(str(tmp_path), {qname}, diff_files)
    assert qname in result


def test_find_outside_callers_no_match(tmp_path):
    outside = tmp_path / "other.py"
    outside.write_text("x = 1\n")
    qname = "mypkg.service.get_user"
    result = _find_outside_callers(str(tmp_path), {qname}, set())
    assert qname not in result


def test_build_alias_map_skips_compound_statement(tmp_path):
    # A function definition is a compound statement, not SimpleStatementLine (line 76).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def helper():\n    pass\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_build_alias_map_skips_non_import_in_simple_stmt(tmp_path):
    # An assignment in SimpleStatementLine is not ImportFrom (line 79).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '1.0'\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_find_outside_callers_call_qname_not_target(tmp_path):
    # outside file calls other_func (resolves to mypkg.other.other_func),
    # but target is mypkg.service.get_user → hits the 118->117 branch.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "other.py").write_text("def other_func(): pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("from mypkg.other import other_func\nother_func()\n")

    result = _find_outside_callers(str(tmp_path), {"mypkg.service.get_user"}, set())
    assert "mypkg.service.get_user" not in result


def test_find_outside_callers_manager_build_fails(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine.FullRepoManager", side_effect=RuntimeError("fail")):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    # Conservative: all target qnames are blocked.
    assert result == {"some.func"}


def test_find_outside_callers_wrapper_fails(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine.FullRepoManager") as MockFRM:
        MockFRM.return_value.get_metadata_wrapper_for_path.side_effect = RuntimeError(
            "fail"
        )
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == set()


def test_cross_file_file_not_under_repo_root(tmp_path):
    # repo_root is a separate directory; changed file is not under it.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    f = tmp_path / "code.py"
    f.write_text("def public_func():\n    return (1, 2, 3)\n", encoding="utf-8")
    # _compute_qname raises ValueError → all_candidates stays empty → 317->406 branch.
    msgs = list(
        run_engine(
            {str(f): [(1, 2)]},
            _repo_root=str(repo_root),
            config=CrispenConfig(min_tuple_size=3),
        )
    )
    assert not any("callers" in m for m in msgs)


def test_no_public_candidates_with_repo_root(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = list(run_engine({str(f): [(1, 1)]}, _repo_root=str(tmp_path)))
    assert msgs == []


def test_cross_file_caller_updater_file_not_under_repo_root(tmp_path):
    subdir = tmp_path / "repo"
    subdir.mkdir()
    (subdir / "__init__.py").write_text("")

    inside = subdir / "service.py"
    inside.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    # This file is in the diff but outside repo_root (subdir).
    outside_code = tmp_path / "outside_code.py"
    outside_code.write_text("x = 1\n", encoding="utf-8")

    changed = {str(inside): [(1, 2)], str(outside_code): [(1, 1)]}
    # No crash; outside_code.py's _file_to_module raises ValueError → continue.
    list(
        run_engine(
            changed, _repo_root=str(subdir), config=CrispenConfig(min_tuple_size=3)
        )
    )


def test_find_outside_callers_scope_analysis_timeout(tmp_path):
    """When _visit_with_timeout times out, all target qnames are blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine._visit_with_timeout", return_value=False):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}


def test_find_outside_callers_deadline_expired(tmp_path):
    """Total budget already exhausted before any file is visited: all blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    # A negative timeout makes the deadline fall in the past immediately.
    with patch("crispen.engine._SCOPE_ANALYSIS_TIMEOUT", -1):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}


def test_find_outside_callers_excludes_venv_dirs(tmp_path):
    """Files inside excluded directories (.venv, __pycache__, etc.) are skipped."""
    for dirname in _EXCLUDED_DIR_NAMES:
        excluded = tmp_path / dirname / "lib"
        excluded.mkdir(parents=True, exist_ok=True)
        (excluded / "pkg.py").write_text(
            "from mypkg.service import get_user\nget_user()\n"
        )
    # Even though each excluded dir has a caller, none should be counted.
    result = _find_outside_callers(str(tmp_path), {"mypkg.service.get_user"}, set())
    assert "mypkg.service.get_user" not in result


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
