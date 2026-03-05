import textwrap
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.file_limiter.runner import FileLimiterResult
from crispen.refactors.base import Refactor
from tests.test_engine_helpers import _write_code_file
from tests.test_tuple_dataclass_td import _FL_PATCH


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
    f, _ = _write_code_file(tmp_path, source)
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
    f, _ = _write_code_file(tmp_path, source)

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
