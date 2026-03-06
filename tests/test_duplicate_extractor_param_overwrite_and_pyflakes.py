from crispen.refactors.duplicate_extractor import (
    _has_param_overwritten_before_read,
    _pyflakes_new_undefined_names,
)


def test_has_param_overwritten_before_read_false_when_param_is_read():
    # Parameter is read before (or without) being reassigned — should return False.
    helper = "def fn(x):\n    return x + 1\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_has_param_overwritten_before_read_true_when_immediately_overwritten():
    # Parameter is assigned on the first statement without being read — True.
    helper = "def setup(client):\n    client = object()\n    return client\n"
    assert _has_param_overwritten_before_read(helper) is True


def test_has_param_overwritten_before_read_false_for_conditional_default():
    # The ``if x is None: x = default`` pattern reads before writing — False.
    helper = "def fn(x=None):\n    if x is None:\n        x = []\n    return x\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_has_param_overwritten_before_read_vararg_and_kwarg():
    # Covers the vararg/kwarg branches — neither is overwritten here.
    helper = "def fn(*args, **kwargs):\n    return args, kwargs\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_pyflakes_new_undefined_names_returns_empty_when_no_new_issues():
    # Names undefined in both original and candidate → no NEW issues.
    original = "def foo():\n    return bar()\n"
    candidate = "def _h():\n    pass\n\ndef foo():\n    return bar()\n"
    assert _pyflakes_new_undefined_names(original, candidate) == set()


def test_pyflakes_new_undefined_names_detects_introduced_name():
    # candidate introduces a reference to an unassigned name not in original.
    original = "def foo():\n    x = 1\n    return x\n"
    # candidate removes the assignment, leaving x undefined at the call site
    candidate = "def _h():\n    x = 1\n\ndef foo():\n    _h(x)\n    return x\n"
    assert "x" in _pyflakes_new_undefined_names(original, candidate)
