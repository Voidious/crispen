from crispen.engine import _categorize_into_stats
from crispen.stats import RunStats


def test_categorize_function_split():
    s = RunStats()
    _categorize_into_stats(s, "split 'big_func': extracted _step_two")
    assert s.function_split == 1


def test_categorize_other_message_ignored():
    s = RunStats()
    _categorize_into_stats(s, "CallerUpdater: expanded FooResult unpacking at line 7")
    assert s.total_edits == 0
