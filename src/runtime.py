"""File-based execution helpers for the SimpleML DSL."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .ast import build_ast_from_parse_tree
from .evaluator import RuntimeState, evaluate_ast
from .parser import ParseError, parse_statement
from .preprocessing import clean_preprocessing_cache


def execute_file(path: str | Path) -> RuntimeState:
    """Parse a DSL script file and execute the statements in order."""

    script_path = Path(path)
    if not script_path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        scripts_path = repo_root / "scripts" / script_path
        if scripts_path.exists():
            script_path = scripts_path
        else:
            raise FileNotFoundError(f"DSL script not found: {script_path}")

    statements = _load_statements(script_path)
    ast_node = build_ast_from_parse_tree(statements)
    state = evaluate_ast(ast_node)
    # Transient preprocessing cache is no longer needed once the run completes.
    clean_preprocessing_cache()
    return state


def _load_statements(path: Path) -> List[object]:
    parsed_statements: List[object] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parsed_statements.append(parse_statement(line))
        except ParseError as exc:
            raise ParseError(f"{path}:{line_number}: {exc}") from exc
    if not parsed_statements:
        raise ParseError(f"No statements found in {path}")
    return parsed_statements
