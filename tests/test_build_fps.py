from crispen.refactors.duplicate_extractor import (
    _FunctionInfo,
    _build_function_body_fps,
    _normalize_source,
)


def _make_func_info(name: str, body_source: str = "    pass\n") -> _FunctionInfo:
    return _FunctionInfo(
        name=name,
        source=f"def {name}():\n{body_source}",
        scope="<module>",
        body_source=body_source,
        body_stmt_count=1,
        params=[],
    )


def test_build_fps_includes_called():
    body = "    x = 1\n    y = 2\n    z = 3\n"
    func = _make_func_info("foo", body)
    fps = _build_function_body_fps([func], {"foo"})
    fp = _normalize_source(body)
    assert fp in fps
    assert fps[fp].name == "foo"


def test_build_fps_excludes_uncalled():
    func = _make_func_info("bar")
    fps = _build_function_body_fps([func], {"foo"})
    assert fps == {}


def test_build_fps_empty_functions():
    fps = _build_function_body_fps([], {"foo"})
    assert fps == {}
