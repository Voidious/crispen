"""Tests for crispen.stats.RunStats."""

from crispen.stats import RunStats


def _filled() -> RunStats:
    s = RunStats()
    s.if_not_else = 2
    s.tuple_to_dataclass = 1
    s.duplicate_extracted = 3
    s.duplicate_matched = 1
    s.function_split = 4
    s.algorithmic_rejected = 0
    s.llm_rejected = 1
    s.llm_veto_calls = 4
    s.llm_edit_calls = 7
    s.llm_verify_calls = 3
    s.files_edited = ["foo.py", "bar.py"]
    s.lines_changed = 45
    return s


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_adds_all_counters():
    a = RunStats(if_not_else=1, tuple_to_dataclass=2, llm_veto_calls=3)
    b = RunStats(if_not_else=10, duplicate_extracted=5, llm_edit_calls=2)
    a.merge(b)
    assert a.if_not_else == 11
    assert a.tuple_to_dataclass == 2
    assert a.duplicate_extracted == 5
    assert a.llm_veto_calls == 3
    assert a.llm_edit_calls == 2


def test_merge_does_not_merge_files_edited():
    a = RunStats()
    a.files_edited = ["a.py"]
    b = RunStats()
    b.files_edited = ["b.py"]
    a.merge(b)
    assert a.files_edited == ["a.py"]


# ---------------------------------------------------------------------------
# property totals
# ---------------------------------------------------------------------------


def test_total_edits():
    s = RunStats(
        if_not_else=2,
        tuple_to_dataclass=1,
        duplicate_extracted=3,
        duplicate_matched=1,
        function_split=4,
    )
    assert s.total_edits == 11


def test_total_rejected():
    s = RunStats(algorithmic_rejected=2, llm_rejected=3)
    assert s.total_rejected == 5


def test_total_llm_calls():
    s = RunStats(llm_veto_calls=4, llm_edit_calls=7, llm_verify_calls=3)
    assert s.total_llm_calls == 14


# ---------------------------------------------------------------------------
# count_lines_changed
# ---------------------------------------------------------------------------


def test_count_lines_changed_added_lines():
    s = RunStats()
    s.count_lines_changed("a = 1\n", "a = 1\nb = 2\n")
    assert s.lines_changed == 1


def test_count_lines_changed_removed_lines():
    s = RunStats()
    s.count_lines_changed("a = 1\nb = 2\n", "a = 1\n")
    assert s.lines_changed == 1


def test_count_lines_changed_no_difference():
    s = RunStats()
    s.count_lines_changed("x = 1\n", "x = 1\n")
    assert s.lines_changed == 0


# ---------------------------------------------------------------------------
# format_summary — with files
# ---------------------------------------------------------------------------


def setup_format_summary_test(stats_factory):
    s = stats_factory()
    lines = s.format_summary()
    text = "\n".join(lines)
    return s, lines, text


def test_format_summary_with_files():
    s, lines, text = setup_format_summary_test(_filled)
    assert "--- crispen summary ---" in text
    assert "if not/else:" in text
    assert "tuple to dataclass:" in text
    assert "duplicate extracted:" in text
    assert "match existing:" in text
    assert "function split:" in text
    assert "total:               11" in text
    assert "algorithmic:" in text
    assert "LLM:" in text
    assert "veto:" in text
    assert "edit:" in text
    assert "verify:" in text
    assert "files edited (2): foo.py, bar.py" in text
    assert "lines changed: 45" in text


# ---------------------------------------------------------------------------
# format_summary — without files (exercises the else branch)
# ---------------------------------------------------------------------------


def test_format_summary_no_files():
    s, lines, text = setup_format_summary_test(RunStats)
    assert "files edited: none" in text
