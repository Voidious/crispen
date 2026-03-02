from crispen.engine import _categorize_into_stats
from crispen.stats import RunStats


def test_categorize_function_split():
    s = RunStats()
    _categorize_into_stats(s, "split 'big_func': extracted _step_two")
    assert s.function_split == 1
