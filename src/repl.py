"""Interactive REPL for the SimpleML DSL."""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

from .ast import build_ast_from_parse_tree
from .evaluator import RuntimeState, evaluate_ast
from .parser import ParseError, parse_statement
from .preprocessing import clean_preprocessing_cache


def clear_screen() -> None:
    """Clear the console screen."""
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")


def show_help() -> None:
    """Show help message listing available commands."""
    help_text = """
SimpleML - Machine Learning DSL

Available commands:
- load <dataset name> from <file path>        Load CSV dataset
- preprocess <dataset name> target <target>  Infer task type and prepare dataset
- train <model name> using <model type> on <dataset>  Train model
- evaluate <model name> using <metrics name>  Evaluate trained model
- predict <model name> with <input data>      Generate predictions
- summary <model name>                        Show pipeline state summary
- clear                                        Clear console screen
- help                                         Show this help message
- exit / quit                                  Exit the REPL

Supported model types:
- linear_regression (regression)
- logistic_regression (classification)
- random_forest (classification/regression)
"""
    print(help_text)


class REPL:
    """A minimal REPL that executes one DSL statement at a time."""

    def __init__(self) -> None:
        self.state = RuntimeState()

    def run(self, input_lines: Iterable[str]) -> RuntimeState:
        for raw_line in input_lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                break
            if line.lower() == "clear":
                clear_screen()
                continue
            if line.lower() == "help":
                show_help()
                continue
            try:
                parsed = parse_statement(line)
                ast_node = build_ast_from_parse_tree(parsed)
                num_datasets_before = len(self.state.datasets)
                self.state = evaluate_ast(ast_node, self.state)
                num_datasets_after = len(self.state.datasets)
            except ParseError as exc:
                print(f"Parse error: {exc}")
            else:
                print(f"Executed: {line}")
                if num_datasets_after > num_datasets_before:
                    print("Loaded dataset")
        return self.state


def run_repl(input_lines: Optional[Iterable[str]] = None) -> RuntimeState:
    """Run the REPL with either an iterable of commands or stdin input."""

    print("SimpleML REPL")
    print("Type 'help' for available commands, 'exit' to quit.")

    repl = REPL()
    if input_lines is None:
        try:
            while True:
                line = input("SimpleML> ")
                if line.strip().lower() in {"exit", "quit"}:
                    break
                repl.run([line])
        except EOFError:
            print()
    else:
        repl.run(input_lines)
    # Transient preprocessing cache is no longer needed once the REPL exits.
    clean_preprocessing_cache()
    return repl.state
