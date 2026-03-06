from __future__ import annotations
from crispen.file_limiter.advisor import _group_summary
from .helpers import _make_entity


def test_group_summary_with_docstring_and_params():
    """Entity with docstring and params → both appear in summary."""
    ent = _make_entity(
        "foo",
        1,
        10,
        docstring="Parse the config file. More details here.",
        params=["path: str", "strict: bool"],
    )
    summary = _group_summary(["foo"], {"foo": ent})
    assert "foo (10 lines)" in summary
    assert '"Parse the config file."' in summary
    assert "params: path: str, strict: bool" in summary


def test_group_summary_with_params_only():
    """Entity with params but no docstring → params appear, no docstring quote."""
    ent = _make_entity("bar", 1, 5, params=["x: int", "y"])
    summary = _group_summary(["bar"], {"bar": ent})
    assert "params: x: int, y" in summary
    assert '"' not in summary


def test_group_summary_docstring_no_period():
    """Docstring with no period → full text used as first sentence."""
    ent = _make_entity("baz", 1, 3, docstring="No period here")
    summary = _group_summary(["baz"], {"baz": ent})
    assert '"No period here"' in summary
