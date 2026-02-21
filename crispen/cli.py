"""CLI entry point: reads stdin diff, drives engine, reports to stdout."""

import sys

from .config import load_config
from .diff_parser import parse_diff
from .engine import run_engine
from .errors import CrispenAPIError


def main() -> None:
    diff_text = sys.stdin.read()
    if not diff_text.strip():
        print("crispen: no diff provided on stdin", file=sys.stderr)
        sys.exit(1)

    changed = parse_diff(diff_text)
    if not changed:
        return

    config = load_config()
    try:
        for message in run_engine(changed, config=config):
            print(message)
    except CrispenAPIError as exc:
        print(f"crispen: {exc}", file=sys.stderr)
        sys.exit(1)
