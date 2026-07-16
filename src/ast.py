"""Abstract syntax tree definitions for the SimpleML DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union, cast


@dataclass(frozen=True)
class LoadStatement:
    dataset_name: str
    source_path: str


@dataclass(frozen=True)
class PreprocessStatement:
    dataset_name: str
    target_name: str


@dataclass(frozen=True)
class TrainStatement:
    model_name: str
    model_type: str
    dataset_name: str


@dataclass(frozen=True)
class PredictStatement:
    model_name: str
    input_data: Any


@dataclass(frozen=True)
class EvaluateStatement:
    model_name: str
    metrics_name: str


@dataclass(frozen=True)
class SummaryStatement:
    target_name: str


@dataclass(frozen=True)
class Pipeline:
    statements: Tuple[StatementNode, ...]


StatementNode = Union[
    LoadStatement,
    PreprocessStatement,
    TrainStatement,
    PredictStatement,
    EvaluateStatement,
    SummaryStatement,
    Pipeline,
]


def _require_string(value: object, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"Invalid parse tree: expected string for {field_name}")


def build_ast_from_parse_tree(tree: object) -> StatementNode:
    """Convert parser tuples into explicit AST nodes.

    The function accepts either a single parse tree for one statement or a
    sequence of parse trees for a pipeline of statements.
    """

    if isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        if not tree:
            raise ValueError("Invalid parse tree")
        if len(tree) == 2 and isinstance(tree[0], str) and tree[0] == "statement":
            return _build_single_statement_ast(tree)
        if all(isinstance(item, tuple) for item in tree):
            statements = tuple(_build_single_statement_ast(item) for item in tree)
            return Pipeline(statements=statements)

    return _build_single_statement_ast(cast(Tuple[object, ...], tree))


def _build_single_statement_ast(tree: Tuple[object, ...]) -> StatementNode:
    if not tree or len(tree) < 2 or tree[0] != "statement":
        raise ValueError("Invalid parse tree")

    payload = cast(Tuple[object, ...], tree[1])
    if not payload or len(payload) < 1 or not isinstance(payload[0], str):
        raise ValueError("Invalid parse tree")

    kind = payload[0]

    if kind == "load":
        if len(payload) != 3:
            raise ValueError("Invalid parse tree for load statement")
        return LoadStatement(
            dataset_name=_require_string(payload[1], field_name="dataset_name"),
            source_path=_require_string(payload[2], field_name="source_path"),
        )
    if kind == "preprocess":
        if len(payload) != 3:
            raise ValueError("Invalid parse tree for preprocess statement")
        return PreprocessStatement(
            dataset_name=_require_string(payload[1], field_name="dataset_name"),
            target_name=_require_string(payload[2], field_name="target_name"),
        )
    if kind == "train":
        if len(payload) != 4:
            raise ValueError("Invalid parse tree for train statement")
        return TrainStatement(
            model_name=_require_string(payload[1], field_name="model_name"),
            model_type=_require_string(payload[2], field_name="model_type"),
            dataset_name=_require_string(payload[3], field_name="dataset_name"),
        )
    if kind == "predict":
        if len(payload) != 3:
            raise ValueError("Invalid parse tree for predict statement")
        return PredictStatement(
            model_name=_require_string(payload[1], field_name="model_name"),
            input_data=payload[2],
        )
    if kind == "evaluate":
        if len(payload) != 3:
            raise ValueError("Invalid parse tree for evaluate statement")
        return EvaluateStatement(
            model_name=_require_string(payload[1], field_name="model_name"),
            metrics_name=_require_string(payload[2], field_name="metrics_name"),
        )
    if kind == "summary":
        if len(payload) != 2:
            raise ValueError("Invalid parse tree for summary statement")
        return SummaryStatement(
            target_name=_require_string(payload[1], field_name="target_name")
        )

    raise ValueError(f"Unsupported statement kind: {kind}")
