"""Tests for the Refactor base class."""

from crispen.refactors.base import Refactor
from crispen.refactors.if_not_else import IfNotElse


def test_line_in_changed_range_true():
    r = Refactor([(1, 10)])
    assert r._line_in_changed_range(5) is True


def test_line_in_changed_range_boundary():
    r = Refactor([(5, 10)])
    assert r._line_in_changed_range(5) is True
    assert r._line_in_changed_range(10) is True


def test_line_in_changed_range_false():
    r = Refactor([(1, 10)])
    assert r._line_in_changed_range(20) is False


def test_line_in_changed_range_false_before_range():
    r = Refactor([(5, 10)])
    assert r._line_in_changed_range(4) is False


def test_name_returns_class_name():
    assert Refactor.name() == "Refactor"
    assert IfNotElse.name() == "IfNotElse"


def test_get_changes_empty():
    r = Refactor([(1, 10)])
    assert list(r.get_changes()) == []


def test_get_changes_after_append():
    r = Refactor([(1, 10)])
    r.changes_made.append("something happened")
    assert list(r.get_changes()) == ["something happened"]
