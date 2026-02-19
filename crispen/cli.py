"""CLI entry point: reads stdin diff, drives engine, reports to stdout."""

import sys

from .diff_parser import parse_diff
from .engine import run_engine


def main() -> None:
    diff_text = sys.stdin.read()
    if not diff_text.strip():
        print("crispen: no diff provided on stdin", file=sys.stderr)
        sys.exit(1)

    changed = parse_diff(diff_text)
    if not changed:
        return

    for message in run_engine(changed):
        print(message)
