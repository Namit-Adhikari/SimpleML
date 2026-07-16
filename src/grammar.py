"""Grammar specification for the SimpleML DSL.

This module defines the minimal command vocabulary for the first implementation
phase. The grammar is intentionally compact and centered on pipeline operations
for tabular CSV-based machine learning workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SUPPORTED_COMMANDS: Tuple[str, ...] = (
    "load",
    "preprocess",
    "train",
    "predict",
    "evaluate",
    "summary",
)

SUPPORTED_TASK_TYPES: Tuple[str, ...] = (
    "classification",
    "regression",
)

SUPPORTED_MODEL_TYPES: Tuple[str, ...] = (
    "linear_regression",
    "logistic_regression",
    "random_forest",
)

SUPPORTED_PREPROCESSING_OPERATIONS: Tuple[str, ...] = (
    "z_score_normalization",
    "one_hot_encoding",
    "label_encoding",
    "duplicate_removal",
    "simple_imputation",
)

SIMPLE_IMPUTATION_BEHAVIOR: Dict[str, str] = {
    "numerical": "median",
    "categorical": "mode",
}

SUPPORTED_EVALUATION_METRICS: Tuple[str, ...] = (
    "accuracy",
    "mae",
    "rmse",
    "r2",
)


@dataclass(frozen=True)
class DSLGrammarSpec:
    """Structured representation of the initial DSL grammar rules."""

    rules: Dict[str, str] = field(
        default_factory=lambda: {
            "statement": "load_statement | preprocess_statement | train_statement | predict_statement | evaluate_statement | summary_statement",
            "load_statement": "load IDENTIFIER FROM STRING",
            "preprocess_statement": "preprocess IDENTIFIER USING STRING",
            "train_statement": "train IDENTIFIER USING MODEL_NAME ON IDENTIFIER",
            "predict_statement": "predict IDENTIFIER USING predict_input",
            "evaluate_statement": "evaluate IDENTIFIER USING IDENTIFIER",
            "summary_statement": "summary IDENTIFIER",
            "model_name": "linear_regression | logistic_regression | random_forest",
            "task_name": "classification | regression",
            "preprocessing_name": "z_score_normalization | one_hot_encoding | label_encoding | duplicate_removal | simple_imputation",
            "predict_input": "dict | list",
        }
    )


def build_ebnf_spec() -> str:
    """Render the grammar as a compact EBNF specification."""

    lines: List[str] = [
        "<statement> ::= <load_statement> | <preprocess_statement> | <train_statement> | <predict_statement> | <evaluate_statement> | <summary_statement>",
        "<load_statement> ::= load <identifier> from <string>",
        "<preprocess_statement> ::= preprocess <identifier> using <string>",
        "<train_statement> ::= train <identifier> using <model_name> on <identifier>",
        "<predict_statement> ::= predict <identifier> using <predict_input>",
        "<evaluate_statement> ::= evaluate <identifier> using <identifier>",
        "<summary_statement> ::= summary <identifier>",
        "<model_name> ::= linear_regression | logistic_regression | random_forest",
        "<task_name> ::= classification | regression",
        "<preprocessing_name> ::= z_score_normalization | one_hot_encoding | label_encoding | duplicate_removal | simple_imputation",
        "<identifier> ::= IDENTIFIER",
        "<string> ::= STRING",
        "<predict_input> ::= <dict> | <list>",
    ]
    return "\n".join(lines)
