"""PLY-based lexer and parser for the SimpleML DSL."""

from __future__ import annotations

import importlib
from typing import Any, Tuple

lex = importlib.import_module("ply.lex")
yacc = importlib.import_module("ply.yacc")

from .grammar import (
    SUPPORTED_COMMANDS,
    SUPPORTED_MODEL_TYPES,
    SUPPORTED_PREPROCESSING_OPERATIONS,
    SUPPORTED_TASK_TYPES,
)


tokens = (
    "IDENTIFIER",
    "STRING",
    "NUMBER",
    "LOAD",
    "PREPROCESS",
    "TRAIN",
    "PREDICT",
    "EVALUATE",
    "SUMMARY",
    "MODEL_NAME",
    "FROM",
    "USING",
    "ON",
    "LBRACE",
    "RBRACE",
    "LBRACKET",
    "RBRACKET",
    "COLON",
    "COMMA",
)

reserved = {
    "load": "LOAD",
    "preprocess": "PREPROCESS",
    "train": "TRAIN",
    "predict": "PREDICT",
    "evaluate": "EVALUATE",
    "summary": "SUMMARY",
}
reserved.update({name: "MODEL_NAME" for name in SUPPORTED_MODEL_TYPES})
reserved.update(
    {
        "from": "FROM",
        "using": "USING",
        "on": "ON",
    }
)


def t_IDENTIFIER(t: Any) -> Any:
    r"[A-Za-z_][A-Za-z0-9_]*"
    t.type = reserved.get(t.value, "IDENTIFIER")
    return t


def t_STRING(t: Any) -> Any:
    r'"([^"\\]|\\.)*"|\'([^\'\\]|\\.)*"'
    t.value = t.value[1:-1]
    return t


def t_NUMBER(t: Any) -> Any:
    r"-?\d+(\.\d+)?([eE][+-]?\d+)?"
    if "." in t.value or "e" in t.value.lower():
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t


t_LBRACE = r"\{"
t_RBRACE = r"\}"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_COLON = r":"
t_COMMA = r","


def t_newline(t: Any) -> None:
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_ignore_whitespace(t: Any) -> None:
    r"\s+"
    return None


def t_error(t: Any) -> None:
    raise SyntaxError(f"Illegal character '{t.value[0]}'")


lexer = lex.lex()


class ParseError(ValueError):
    """Raised when a DSL statement cannot be parsed."""


def p_statement(p):
    """statement : load_statement
    | preprocess_statement
    | train_statement
    | predict_statement
    | evaluate_statement
    | summary_statement"""
    p[0] = ("statement", p[1])


def p_load_statement(p):
    """load_statement : LOAD IDENTIFIER FROM STRING
    | LOAD IDENTIFIER FROM IDENTIFIER"""
    p[0] = ("load", p[2], p[4])


def p_preprocess_statement(p):
    """preprocess_statement : PREPROCESS IDENTIFIER USING STRING
    | PREPROCESS IDENTIFIER USING IDENTIFIER"""
    p[0] = ("preprocess", p[2], p[4])


def p_train_statement(p):
    """train_statement : TRAIN IDENTIFIER USING MODEL_NAME ON IDENTIFIER"""
    p[0] = ("train", p[2], p[4], p[6])


def p_predict_statement(p):
    """predict_statement : PREDICT IDENTIFIER USING predict_input"""
    p[0] = ("predict", p[2], p[4])


def p_predict_input_dict(p):
    """predict_input : dict"""
    p[0] = p[1]


def p_predict_input_list(p):
    """predict_input : list"""
    p[0] = p[1]


def p_predict_input_identifier(p):
    """predict_input : IDENTIFIER"""
    p[0] = p[1]


def p_predict_input_string(p):
    """predict_input : STRING"""
    p[0] = p[1]


def p_dict(p):
    """dict : LBRACE key_value_list RBRACE"""
    p[0] = dict(p[2])


def p_dict_empty(p):
    """dict : LBRACE RBRACE"""
    p[0] = {}


def p_key_value_list(p):
    """key_value_list : key_value
    | key_value_list COMMA key_value"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_key_value(p):
    """key_value : key COLON value"""
    p[0] = (p[1], p[3])


def p_key_identifier(p):
    """key : IDENTIFIER"""
    p[0] = p[1]


def p_key_string(p):
    """key : STRING"""
    p[0] = p[1]


def p_value_number(p):
    """value : NUMBER"""
    p[0] = p[1]


def p_value_string(p):
    """value : STRING"""
    p[0] = p[1]


def p_list(p):
    """list : LBRACKET value_list RBRACKET"""
    p[0] = p[2]


def p_list_empty(p):
    """list : LBRACKET RBRACKET"""
    p[0] = []


def p_value_list(p):
    """value_list : value
    | value_list COMMA value"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_value_dict(p):
    """value : dict"""
    p[0] = p[1]


def p_value_list_as_value(p):
    """value : list"""
    p[0] = p[1]


def p_evaluate_statement(p):
    """evaluate_statement : EVALUATE IDENTIFIER USING IDENTIFIER"""
    p[0] = ("evaluate", p[2], p[4])


def p_summary_statement(p):
    """summary_statement : SUMMARY IDENTIFIER"""
    p[0] = ("summary", p[2])


def p_error(p):
    if p is None:
        raise ParseError("Unexpected end of input")
    raise ParseError(f"Syntax error at token {p.value!r}")


parser = yacc.yacc(debug=False, write_tables=False)


def parse_statement(source: str) -> Tuple[object, ...]:
    """Parse a single DSL statement into a structured tuple."""

    if not source.strip():
        raise ParseError("Empty input")

    lexer.input(source)
    result = parser.parse(source, lexer=lexer)
    return result
