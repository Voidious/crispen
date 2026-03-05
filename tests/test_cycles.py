from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind
from tests.test_helpers import _classified, _make_entity, _plan


def test_generate_aborts_on_cross_file_import_cycle():
    # fn_a references fn_b (in b.py) and fn_b references fn_a (in a.py).
    # This creates a circular import a.py ↔ b.py that Python cannot load.
    # generate_file_splits must detect the cycle and abort rather than emit
    # broken code.
    source = "def fn_a():\n    return fn_b()\n\ndef fn_b():\n    return fn_a()\n"
    e_a = _make_entity("fn_a", 1, 2)
    e_b = _make_entity("fn_b", 4, 5)
    c = _classified(entities=[e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["fn_a"], target_file="a.py"),
            GroupPlacement(group=["fn_b"], target_file="b.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.abort is True
    assert result.new_files == {}


def test_generate_aborts_on_cycle_through_original():
    # _CONST is a TOP_LEVEL constant (stays in original).
    # _worker is migrated to helpers.py and references _CONST.
    # main() (non-migrated) calls _worker → original will re-export _worker.
    # Cycle: original → helpers.py (re-export of _worker)
    #              → original (via `from .original import _CONST`).
    source = textwrap.dedent(
        """\
        _CONST = "value"

        def _worker():
            return _CONST

        def main():
            return _worker()
    """
    )
    e_const = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_worker = _make_entity("_worker", 3, 4)
    e_main = _make_entity("main", 6, 7)
    c = _classified(entities=[e_const, e_worker, e_main])
    plan = _plan([GroupPlacement(group=["_worker"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    # helpers.py would need `from .original import _CONST` while original
    # re-exports _worker from helpers.py → circular import → must abort.
    assert result.abort is True
    assert result.new_files == {}
