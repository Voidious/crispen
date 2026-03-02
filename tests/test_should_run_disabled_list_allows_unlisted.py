from crispen.config import CrispenConfig
from crispen.engine import _should_run


def test_should_run_disabled_list_allows_unlisted():
    cfg = CrispenConfig(disabled_refactors=["function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("tuple_dataclass", cfg) is True
