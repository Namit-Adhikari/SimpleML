"""CLI entry point for the SimpleML DSL, package-internal implementation."""

from __future__ import annotations

import argparse
import sys

from .repl import run_repl
from .runtime import execute_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SimpleML DSL for tabular machine learning pipelines"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a DSL script file")
    run_parser.add_argument(
        "script",
        help="Path or bare name of a .dsl script (resolved from scripts/)",
    )

    subparsers.add_parser("repl", help="Start the interactive REPL")

    args = parser.parse_args(argv)

    if args.command == "run":
        state = execute_file(args.script)
        for summary in state.summaries:
            print(summary)
        return 0

    if args.command == "repl":
        run_repl()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
