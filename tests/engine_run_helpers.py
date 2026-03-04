from crispen.config import CrispenConfig
from crispen.engine import run_engine


def _run_engine_expect_outside_callers_msg(service, tmp_path):
    changed = {str(service): [(1, 2)]}
    msgs = list(
        run_engine(
            changed,
            _repo_root=str(tmp_path),
            config=CrispenConfig(min_tuple_size=3),
        )
    )
    assert any("callers exist outside the diff" in m for m in msgs)
    return msgs


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))
