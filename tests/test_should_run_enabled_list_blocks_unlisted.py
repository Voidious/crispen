from crispen.config import CrispenConfig
from crispen.engine import _should_run


def test_should_run_enabled_list_blocks_unlisted():
    cfg = CrispenConfig(enabled_refactors=["if_not_else"])
    assert _should_run("duplicate_extractor", cfg) is False
    assert _should_run("tuple_dataclass", cfg) is False
    assert _should_run("file_limiter", cfg) is False
