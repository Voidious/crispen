from crispen.file_limiter.runner import FileLimiterResult


def _make_fl_result_with_entities(source="# reduced\n"):
    """Build a FileLimiterResult that moved MyClass → utils.py."""
    return FileLimiterResult(
        original_source=source,
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=["big.py: FileLimiter: moved MyClass → utils.py"],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )


_REWRITE_PATCH = "crispen.engine.apply_patch_rewrite"


_CG_PATCH = "crispen.engine.apply_patch_callgraph"
