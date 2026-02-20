"""Load files, apply refactors, verify, and write back."""

from pathlib import Path
from typing import Dict, Generator, List, Tuple

import libcst as cst
from libcst.metadata import MetadataWrapper

from .errors import CrispenAPIError
from .refactors.duplicate_extractor import DuplicateExtractor
from .refactors.if_not_else import IfNotElse
from .refactors.tuple_dataclass import TupleDataclass

_REFACTORS = [IfNotElse, TupleDataclass, DuplicateExtractor]


def run_engine(
    changed: Dict[str, List[Tuple[int, int]]],
    verbose: bool = True,
) -> Generator[str, None, None]:
    """Apply all refactors to changed files and yield summary messages."""
    for filepath, ranges in changed.items():
        path = Path(filepath)
        if not path.exists():
            yield f"SKIP {filepath}: file not found"
            continue

        source = path.read_text(encoding="utf-8")
        modified = False
        current_source = source

        for RefactorClass in _REFACTORS:
            try:
                current_tree = cst.parse_module(current_source)
            except cst.ParserSyntaxError as exc:
                yield f"SKIP {filepath} ({RefactorClass.name()}): parse error: {exc}"
                break

            wrapper = MetadataWrapper(current_tree)
            try:
                transformer = RefactorClass(
                    ranges, source=current_source, verbose=verbose
                )
                new_tree = wrapper.visit(transformer)
            except CrispenAPIError:
                raise
            except Exception as exc:
                name = RefactorClass.name()
                yield f"SKIP {filepath} ({name}): transform error: {exc}"
                continue

            new_source = new_tree.code
            if new_source == current_source:
                continue

            # Verify output is valid Python
            try:
                compile(new_source, filepath, "exec")
            except SyntaxError as exc:  # pragma: no cover
                name = RefactorClass.name()
                yield f"SKIP {filepath} ({name}): output not valid Python: {exc}"
                continue

            for msg in transformer.get_changes():
                yield f"{filepath}: {msg}"

            current_source = new_source
            modified = True

        if modified:
            path.write_text(current_source, encoding="utf-8")
