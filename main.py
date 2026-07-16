"""CLI entry point wrapper for the SimpleML DSL."""

from __future__ import annotations

import sys

try:
    from simpleml.cli import main
except ImportError:
    # Fallback for running from source without installation
    from src.cli import main


if __name__ == "__main__":
    sys.exit(main())
