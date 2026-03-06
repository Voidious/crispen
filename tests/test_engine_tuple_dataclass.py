from unittest.mock import patch
import pytest
from crispen.config import CrispenConfig
from crispen.engine import _apply_tuple_dataclass, run_engine
from crispen.errors import CrispenAPIError
from tests.test_engine_cross_file_analysis import _make_pkg


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


def test_phase2_apply_tuple_dataclass_td_none(tmp_path):
    """Phase 2 _apply_tuple_dataclass returning td=None is handled gracefully."""
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    orig_apply = _apply_tuple_dataclass
    call_count = {"n": 0}

    def patched_apply(filepath, ranges, source, verbose, approved_public_funcs, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Phase 2 call: return td=None to exercise the td2 is None branch
            return (source, [], None)
        return orig_apply(
            filepath, ranges, source, verbose, approved_public_funcs, **kw
        )

    with patch("crispen.engine._apply_tuple_dataclass", patched_apply):
        msgs = list(
            run_engine(
                {str(service): [(1, 2)]},
                _repo_root=str(tmp_path),
                config=CrispenConfig(min_tuple_size=3),
            )
        )
    # Should not crash; Phase 2 gracefully skips categorization
    assert isinstance(msgs, list)
